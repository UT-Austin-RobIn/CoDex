"""Point-backend boundary checks without paid requests or model downloads."""

import json
import math
import os
from contextlib import nullcontext
from types import SimpleNamespace
import unittest
from unittest import mock

from PIL import Image

from codex.keypoint_estimation.pointing import (
    MolmoPointPointing,
    SolPointing,
    create_point_backend,
)


class PointingTests(unittest.TestCase):
    def setUp(self):
        self.image = Image.new("RGB", (200, 100), "white")

    def _sol(self, payload=None, response=None):
        if response is None:
            response = SimpleNamespace(
                output_text=json.dumps(payload), status="completed", output=[],
                model="gpt-6-sol", usage=None,
            )
        client = SimpleNamespace(responses=SimpleNamespace(create=mock.Mock(return_value=response)))
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}), mock.patch(
            "openai.OpenAI", return_value=client
        ) as constructor:
            backend = create_point_backend()
        self.assertIsInstance(backend, SolPointing)
        self.assertEqual(constructor.call_args.kwargs["timeout"], 120.0)
        self.assertEqual(constructor.call_args.kwargs["max_retries"], 2)
        return backend, client.responses.create

    def test_sol_uses_one_structured_request_for_two_points(self):
        backend, create = self._sol({
            "fn": {"visible": True, "x": 12.5, "y": 75},
            "act": {"visible": False, "x": None, "y": None},
        })
        answer = backend.predict(self.image, "spray plant", "spray bottle", {
            "fn": "liquid outlet that sprays target", "act": "surface pressed to spray",
        })
        self.assertEqual(answer["fn"]["point"], [12.5, 75.0])
        self.assertEqual(answer["act"]["point"], None)
        self.assertFalse(answer["act"]["visible"])
        self.assertEqual(create.call_count, 1)
        args = create.call_args.kwargs
        self.assertEqual(args["model"], "gpt-6-sol")
        self.assertEqual(args["reasoning"], {"effort": "high"})
        self.assertEqual(args["max_output_tokens"], 4096)
        self.assertEqual(args["input"][0]["content"][1]["detail"], "high")
        self.assertTrue(args["input"][0]["content"][1]["image_url"].startswith("data:image/png;base64,"))
        schema = args["text"]["format"]
        self.assertTrue(schema["strict"])
        self.assertEqual(set(schema["schema"]["required"]), {"fn", "act"})
        for kind in ("fn", "act"):
            for axis in ("x", "y"):
                coordinate = schema["schema"]["properties"][kind]["properties"][axis]
                self.assertEqual(coordinate, {"type": ["number", "null"], "minimum": 0, "maximum": 100})
        prompt = args["input"][0]["content"][0]["text"]
        self.assertIn("outlet", prompt)
        self.assertIn("press", prompt)
        self.assertIn("spray plant", prompt)

    def test_sol_rejects_incoherent_and_nonfinite_answers(self):
        bad = [
            {"visible": True, "x": None, "y": 50},
            {"visible": False, "x": 25, "y": 50},
            {"visible": True, "x": 101, "y": 50},
            {"visible": True, "x": float("nan"), "y": 50},
            {"visible": True, "x": "25", "y": 50},
            {"visible": True, "x": True, "y": 50},
        ]
        for point in bad:
            with self.subTest(point=point):
                backend, _ = self._sol({"target": point})
                with self.assertRaises(ValueError):
                    backend.predict(self.image, "", "plant", {"target": "place where spray lands"})

    def test_out_of_bounds_error_identifies_returned_coordinate(self):
        backend, _ = self._sol({"fn": {"visible": True, "x": 101.25, "y": 50}})
        with self.assertRaisesRegex(ValueError, r"fn\.x.*101\.25.*0\.\.100"):
            backend.predict(self.image, "spray plant", "bottle", {"fn": "liquid outlet"})

    def test_sol_refusal_and_missing_output_fail_closed(self):
        for response in [
            SimpleNamespace(status="incomplete", output_text="", output=[], model="gpt-6-sol"),
            SimpleNamespace(status="completed", output_text="", output=[], model="gpt-6-sol"),
            SimpleNamespace(status="completed", output_text="{}", output=[], model="gpt-6-sol"),
        ]:
            with self.subTest(response=response):
                backend, _ = self._sol(response=response)
                with self.assertRaises((RuntimeError, ValueError)):
                    backend.predict(self.image, "", "plant", {"target": "where spray lands"})

    def test_sol_requires_key_only_when_constructed_and_supports_model_override(self):
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": ""}):
            with self.assertRaisesRegex(RuntimeError, "OPENAI_API_KEY"):
                create_point_backend()
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test-key", "CODEX_GPT_MODEL": "chosen"}), mock.patch("openai.OpenAI"):
            self.assertEqual(create_point_backend().model, "chosen")

    def test_molmopoint_pixel_rows_use_image_width_and_height(self):
        self.assertEqual(MolmoPointPointing._point_from_rows([["point", 50, 75]], (200, 100)), [25.0, 75.0])
        self.assertIsNone(MolmoPointPointing._point_from_rows([], (200, 100)))
        for rows in [[[1, 2], [3, 4]], [[201, 50]], [[math.inf, 40]], [["bad", 30]]]:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                MolmoPointPointing._point_from_rows(rows, (200, 100))

    def test_molmopoint_queries_each_kind_and_respects_empty_native_points(self):
        backend = MolmoPointPointing.__new__(MolmoPointPointing)
        backend._query = mock.Mock(side_effect=[("native text", [["point", 50, 75]]), ("not visible", [])])
        result = backend.predict(self.image, "spray plant", "spray bottle", {
            "fn": "liquid outlet", "act": "button pressed by finger",
        })
        self.assertEqual(result["fn"]["point"], [25.0, 75.0])
        self.assertIsNone(result["act"]["point"])
        self.assertEqual(backend._query.call_count, 2)

    def test_molmopoint_passes_processor_metadata_to_native_decoder(self):
        class Tensor:
            def to(self, device):
                self.device = device
                return self

            def size(self, dimension):
                self_dimension = dimension
                return 4 if self_dimension == 1 else 1

            def __getitem__(self, key):
                return self

        tokens = Tensor()
        processor = SimpleNamespace(
            apply_chat_template=mock.Mock(return_value={
                "metadata": {"token_pooling": "pool", "subpatch_mapping": "map", "image_sizes": [(200, 100)]},
                "input_ids": tokens,
            }),
            post_process_image_text_to_text=mock.Mock(return_value=["native output"]),
        )
        model = SimpleNamespace(
            build_logit_processor_from_inputs=mock.Mock(return_value="logits"),
            generate=mock.Mock(return_value=tokens),
            extract_image_points=mock.Mock(return_value=[[0, 0, 50, 75]]),
        )
        backend = MolmoPointPointing.__new__(MolmoPointPointing)
        backend._processor = processor
        backend._model = model
        backend._device = "cpu"
        backend._torch = SimpleNamespace(inference_mode=nullcontext)
        raw, points = backend._query(self.image, "find the outlet")
        self.assertEqual(raw, "native output")
        self.assertEqual(points, [[0, 0, 50, 75]])
        model.extract_image_points.assert_called_once_with("native output", "pool", "map", [(200, 100)])

    def test_molmopoint_reports_old_runtime_before_loading_model(self):
        with mock.patch("codex.keypoint_estimation.pointing.metadata.version", return_value="2.5.1"):
            with self.assertRaisesRegex(RuntimeError, "torch>=2.6.*optional MolmoPoint environment"):
                create_point_backend("molmopoint")


if __name__ == "__main__":
    unittest.main()
