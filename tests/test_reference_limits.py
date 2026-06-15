import unittest

from pipelines.script2video_pipeline import _max_reference_images, _normalize_reference_selector_output
from tools.image_generator_dashscope_api import ImageGeneratorDashScopeAPI


class _LimitedImageGenerator:
    max_reference_images = 2


class ReferenceLimitTests(unittest.TestCase):
    def test_dashscope_image_generator_declares_reference_limit(self):
        generator = ImageGeneratorDashScopeAPI(api_key="test-key")

        self.assertEqual(_max_reference_images(generator), 3)

    def test_selector_output_is_trimmed_to_image_generator_limit(self):
        selector_output = {
            "reference_image_path_and_text_pairs": [
                ("image-0.png", "first reference"),
                ("image-1.png", "second reference"),
                ("image-2.png", "third reference"),
            ],
            "text_prompt": "Use Image 2 as the main background.",
        }

        normalized = _normalize_reference_selector_output(
            selector_output,
            frame_description="A close-up portrait in a rainy street.",
            max_reference_images=_max_reference_images(_LimitedImageGenerator()),
        )

        self.assertEqual(len(normalized["reference_image_path_and_text_pairs"]), 2)
        self.assertNotIn("Image 2", normalized["text_prompt"])
        self.assertIn("provided reference images", normalized["text_prompt"])


if __name__ == "__main__":
    unittest.main()
