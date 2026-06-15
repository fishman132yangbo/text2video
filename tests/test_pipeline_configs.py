from pathlib import Path
import unittest

import yaml


class PipelineConfigTests(unittest.TestCase):
    def test_default_pipeline_configs_use_dashscope_generators(self) -> None:
        for config_path in (Path("configs/idea2video.yaml"), Path("configs/script2video.yaml")):
            with self.subTest(config=str(config_path)):
                config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

                self.assertEqual(config["chat_model"]["init_args"]["model"], "qwen-plus")
                self.assertEqual(
                    config["chat_model"]["init_args"]["base_url"],
                    "https://dashscope.aliyuncs.com/compatible-mode/v1",
                )
                self.assertEqual(config["image_generator"]["class_path"], "tools.ImageGeneratorDashScopeAPI")
                self.assertEqual(config["image_generator"]["init_args"]["model"], "qwen-image")
                self.assertEqual(config["image_generator"]["init_args"]["base_url"], "https://dashscope.aliyuncs.com")
                self.assertEqual(config["video_generator"]["class_path"], "tools.VideoGeneratorDashScopeAPI")
                self.assertEqual(config["video_generator"]["init_args"]["t2v_model"], "wanx2.1-t2v-turbo")


if __name__ == "__main__":
    unittest.main()
