const form = document.querySelector("#jobForm");
const modeButtons = [...document.querySelectorAll(".mode-button")];
const ideaField = document.querySelector("#ideaField");
const scriptField = document.querySelector("#scriptField");
const ideaInput = document.querySelector("#ideaInput");
const scriptInput = document.querySelector("#scriptInput");
const requirementInput = document.querySelector("#requirementInput");
const styleInput = document.querySelector("#styleInput");
const submitButton = document.querySelector("#submitButton");
const connectionStatus = document.querySelector("#connectionStatus");
const jobTitle = document.querySelector("#jobTitle");
const jobState = document.querySelector("#jobState");
const progressFill = document.querySelector("#progressFill");
const timeline = document.querySelector("#timeline");
const logStream = document.querySelector("#logStream");
const videoWrap = document.querySelector("#videoWrap");
const videoPlayer = document.querySelector("#videoPlayer");
const downloadLink = document.querySelector("#downloadLink");
const clearLogButton = document.querySelector("#clearLogButton");

let currentMode = "idea2video";
let activeSource = null;
let eventCount = 0;

const terminalEvents = new Set(["completed", "failed"]);
const stateLabels = {
  idle: "待命",
  queued: "排队中",
  running: "生成中",
  completed: "已完成",
  failed: "失败",
};
const typeLabels = {
  job: "任务",
  queued: "排队",
  running: "运行",
  progress: "进度",
  completed: "完成",
  failed: "失败",
  error: "错误",
};
const messageLabels = {
  "idea is required for idea2video jobs": "请填写创意内容",
  "script is required for script2video jobs": "请填写剧本内容",
  "request body must be JSON": "请求内容必须是 JSON",
  "Job queued": "任务已排队",
  "Job started": "任务已启动",
  "Video generation completed": "视频生成完成",
  "Loading Idea2Video configuration": "正在加载创意生成配置",
  "Loading Script2Video configuration": "正在加载剧本生成配置",
  "Starting Idea2Video pipeline": "正在启动创意生成流程",
  "Starting Script2Video pipeline": "正在启动剧本生成流程",
};

modeButtons.forEach((button) => {
  button.addEventListener("click", () => {
    currentMode = button.dataset.mode;
    modeButtons.forEach((item) => item.classList.toggle("active", item === button));
    ideaField.classList.toggle("hidden", currentMode !== "idea2video");
    scriptField.classList.toggle("hidden", currentMode !== "script2video");
  });
});

clearLogButton.addEventListener("click", () => {
  logStream.replaceChildren();
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  closeEventSource();
  resetJobView();

  const payload = {
    mode: currentMode,
    user_requirement: requirementInput.value,
    style: styleInput.value,
  };
  if (currentMode === "idea2video") {
    payload.idea = ideaInput.value;
  } else {
    payload.script = scriptInput.value;
  }

  submitButton.disabled = true;
  connectionStatus.textContent = "提交中";

  try {
    const response = await fetch("/api/jobs", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(payload),
    });
    const body = await response.json();
    if (!response.ok) {
      throw new Error(body.error || `Request failed with ${response.status}`);
    }
    renderSnapshot(body);
    appendLog("job", `已创建任务 ${body.job_id}`);
    subscribeToJob(body.job_id);
  } catch (error) {
    setState("failed");
    connectionStatus.textContent = "错误";
    appendLog("error", localizeMessage(error.message));
    submitButton.disabled = false;
  }
});

function subscribeToJob(jobId) {
  activeSource = new EventSource(`/api/jobs/${jobId}/events`);
  connectionStatus.textContent = "接收进度";

  activeSource.addEventListener("queued", (event) => consumeEvent(jobId, event));
  activeSource.addEventListener("running", (event) => consumeEvent(jobId, event));
  activeSource.addEventListener("progress", (event) => consumeEvent(jobId, event));
  activeSource.addEventListener("completed", (event) => consumeEvent(jobId, event));
  activeSource.addEventListener("failed", (event) => consumeEvent(jobId, event));
  activeSource.onerror = () => {
    if (submitButton.disabled) {
      connectionStatus.textContent = "正在重连";
    }
  };
}

function consumeEvent(jobId, event) {
  const payload = JSON.parse(event.data);
  eventCount += 1;
  renderEvent(payload);
  appendLog(payload.type, localizeMessage(payload.message), payload.metadata);

  if (terminalEvents.has(payload.type)) {
    closeEventSource();
    submitButton.disabled = false;
    refreshJob(jobId);
  }
}

async function refreshJob(jobId) {
  const response = await fetch(`/api/jobs/${jobId}`);
  const body = await response.json();
  renderSnapshot(body);
}

function renderSnapshot(snapshot) {
  jobTitle.textContent = snapshot.job_id ? `任务 ${snapshot.job_id}` : "暂无任务";
  setState(snapshot.status || "idle");
  if (snapshot.video_ready && snapshot.job_id) {
    const videoUrl = `/api/jobs/${snapshot.job_id}/video`;
    videoPlayer.src = videoUrl;
    downloadLink.href = videoUrl;
    videoWrap.classList.remove("hidden");
  }
}

function renderEvent(event) {
  const item = document.createElement("li");
  const body = document.createElement("div");
  const title = document.createElement("strong");
  const detail = document.createElement("span");
  const stage = event.metadata?.stage || event.type;

  title.textContent = stage;
  detail.textContent = localizeMessage(event.message);
  body.append(title, detail);
  item.append(body);
  item.className = event.type === "completed" ? "done" : event.type === "failed" ? "active" : "active";
  timeline.append(item);
  timeline.scrollTop = timeline.scrollHeight;

  const width = event.type === "completed" ? 100 : Math.min(92, 12 + eventCount * 9);
  progressFill.style.width = `${width}%`;
  setState(event.type === "failed" ? "failed" : event.type === "completed" ? "completed" : "running");
}

function appendLog(type, message, metadata = {}) {
  const line = document.createElement("div");
  line.className = "log-line";
  const time = new Date().toLocaleTimeString();
  const suffix = formatMetadata(metadata);
  line.innerHTML = `<span class="log-time">${escapeHtml(time)}</span> ${escapeHtml(localizeType(type))} ${escapeHtml(message || "")}${escapeHtml(suffix)}`;
  logStream.append(line);
  logStream.scrollTop = logStream.scrollHeight;
}

function formatMetadata(metadata = {}) {
  if (!metadata || !Object.keys(metadata).length) return "";

  const parts = [];
  if (Array.isArray(metadata.possible_causes) && metadata.possible_causes.length) {
    parts.push(`可能原因：${metadata.possible_causes.join("；")}`);
  }
  if (Array.isArray(metadata.evidence) && metadata.evidence.length) {
    parts.push(`当前证据：${metadata.evidence.join("；")}`);
  }
  if (Array.isArray(metadata.next_steps) && metadata.next_steps.length) {
    parts.push(`处理建议：${metadata.next_steps.join("；")}`);
  }

  const rest = {...metadata};
  delete rest.message;
  delete rest.possible_causes;
  delete rest.evidence;
  delete rest.next_steps;
  if (Object.keys(rest).length) {
    parts.push(JSON.stringify(rest));
  }

  return parts.length ? ` ${parts.join(" ")}` : "";
}

function setState(state) {
  jobState.textContent = stateLabels[state] || state;
  jobState.className = "job-state";
  if (state === "running" || state === "queued") jobState.classList.add("state-running");
  if (state === "completed") jobState.classList.add("state-completed");
  if (state === "failed") jobState.classList.add("state-failed");
}

function resetJobView() {
  eventCount = 0;
  timeline.replaceChildren();
  progressFill.style.width = "0";
  videoWrap.classList.add("hidden");
  videoPlayer.removeAttribute("src");
  downloadLink.removeAttribute("href");
  jobTitle.textContent = "正在创建任务";
  setState("queued");
}

function closeEventSource() {
  if (activeSource) {
    activeSource.close();
    activeSource = null;
  }
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function localizeType(type) {
  return typeLabels[type] || type;
}

function localizeMessage(message) {
  return messageLabels[message] || message;
}
