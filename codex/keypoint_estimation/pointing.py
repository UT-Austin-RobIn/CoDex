"""Task keypoints from GPT-6 Sol or MolmoPoint, in image percentages.

The OpenAI client is safe to share across view workers. MolmoPoint owns one
local model and is intended to be called sequentially.
"""

from __future__ import annotations

import base64
from contextlib import nullcontext
from importlib import metadata
from io import BytesIO
import json
import math
import os
import re

from PIL import Image


_MEANINGS = {
    "fn": "The visible functional outlet or contact surface of the functional object that acts on the task target. Locate the working surface, not the handle or control.",
    "act": "The visible surface a hand or finger contacts to operate the object for this task, such as a button, trigger, lever, or grip. Locate the operating contact, not the functional outlet.",
    "target": "The visible point on the target object where the functional surface should act to complete the task.",
}


def _validate_request(image, task, object_name, descriptions):
    if not isinstance(image, Image.Image) or image.width <= 0 or image.height <= 0:
        raise ValueError("image must be a nonempty PIL image")
    if not isinstance(task, str) or not isinstance(object_name, str):
        raise ValueError("task and object_name must be strings")
    if not isinstance(descriptions, dict) or not descriptions:
        raise ValueError("descriptions must be a nonempty mapping")
    for kind, description in descriptions.items():
        if kind not in _MEANINGS:
            raise ValueError(f"unknown point kind: {kind!r}")
        if not isinstance(description, str) or not description.strip():
            raise ValueError(f"description for {kind} must be nonempty")


def _prompt(task, object_name, descriptions):
    lines = [
        f"Object: {object_name}. Task: {task}.",
        "Locate each requested point independently in the supplied image:",
    ]
    for kind, description in descriptions.items():
        lines.append(f"{kind}: {_MEANINGS[kind]} Specific guidance: {description}.")
    lines.extend([
        "For each kind, return exactly one point only if its surface is directly visible and identifiable.",
        "If hidden, occluded, or uncertain, set visible=false and x=y=null. Do not guess through the object or substitute a nearby part.",
        "Coordinates are percentages of the whole image: x=0 left and x=100 right; y=0 top and y=100 bottom.",
        "For a visible surface, point to its center.",
    ])
    return "\n".join(lines)


def _percentage(value, kind, axis):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{kind}.{axis}={value!r} must be a finite number")
    number = float(value)
    if not math.isfinite(number) or not 0 <= number <= 100:
        raise ValueError(f"{kind}.{axis}={value!r} must be within 0..100")
    return number


def _validated_record(kind, record, description):
    if not isinstance(record, dict) or set(record) != {"visible", "x", "y"}:
        raise ValueError(f"{kind} must contain visible, x, and y")
    visible = record["visible"]
    if not isinstance(visible, bool):
        raise ValueError(f"{kind}.visible must be a boolean")
    if visible:
        point = [_percentage(record["x"], kind, "x"), _percentage(record["y"], kind, "y")]
    else:
        if record["x"] is not None or record["y"] is not None:
            raise ValueError(f"{kind} invisible point must have null coordinates")
        point = None
    return {"visible": visible, "point": point, "description": description, "raw": record}


class SolPointing:
    """One strict Responses API request for all requested point kinds."""

    concurrent_safe = True

    def __init__(self, model=None):
        from codex.utils.api_keys import get_key
        api_key = get_key("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set; export it or save it in the app to use the Sol point backend")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Sol pointing requires the openai package") from exc
        self.client = OpenAI(api_key=api_key, timeout=120.0, max_retries=2)
        self.model = model or os.environ.get("CODEX_GPT_MODEL") or "gpt-6-sol"

    def predict(self, image: Image.Image, task: str, object_name: str,
                descriptions: dict[str, str]) -> dict:
        _validate_request(image, task, object_name, descriptions)
        point_schema = {
            "type": "object",
            "properties": {
                "visible": {"type": "boolean"},
                "x": {"type": ["number", "null"], "minimum": 0, "maximum": 100},
                "y": {"type": ["number", "null"], "minimum": 0, "maximum": 100},
            },
            "required": ["visible", "x", "y"],
            "additionalProperties": False,
        }
        schema = {
            "type": "object", "properties": {kind: point_schema for kind in descriptions},
            "required": list(descriptions), "additionalProperties": False,
        }
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        image_url = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
        response = self.client.responses.create(
            model=self.model,
            reasoning={"effort": "high"},
            input=[{"role": "user", "content": [
                {"type": "input_text", "text": _prompt(task, object_name, descriptions)},
                {"type": "input_image", "image_url": image_url, "detail": "high"},
            ]}],
            text={"format": {"type": "json_schema", "name": "task_keypoints",
                             "strict": True, "schema": schema}},
            max_output_tokens=4096,
        )
        if getattr(response, "status", None) != "completed":
            raise RuntimeError(f"Sol point request did not complete: {getattr(response, 'status', None)}")
        output_text = getattr(response, "output_text", None)
        if not output_text:
            raise RuntimeError("Sol point request returned no structured output (possibly refused)")
        try:
            data = json.loads(output_text)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("Sol point request returned invalid JSON") from exc
        if not isinstance(data, dict) or set(data) != set(descriptions):
            raise ValueError("Sol point request returned incorrect point kinds")
        result = {kind: _validated_record(kind, data[kind], description)
                  for kind, description in descriptions.items()}
        model = getattr(response, "model", None)
        usage = getattr(response, "usage", None)
        for record in result.values():
            if model:
                record["model"] = model
            if usage is not None:
                record["usage"] = usage.model_dump() if hasattr(usage, "model_dump") else usage
        return result


def _version_at_least(package, minimum):
    try:
        found = metadata.version(package)
    except metadata.PackageNotFoundError as exc:
        raise RuntimeError(
            f"MolmoPoint requires {package}>={minimum}; run it in the optional MolmoPoint environment"
        ) from exc
    numbers = tuple(int(part) for part in re.match(r"\d+(?:\.\d+)*", found).group().split("."))
    expected = tuple(int(part) for part in minimum.split("."))
    if numbers < expected:
        raise RuntimeError(
            f"MolmoPoint requires {package}>={minimum} (found {found}); run it in the optional MolmoPoint environment"
        )


class MolmoPointPointing:
    """Local native pointing decoder; constructed only when selected."""

    concurrent_safe = False

    def __init__(self, model_name=None):
        _version_at_least("torch", "2.6")
        _version_at_least("transformers", "4.57.1")
        try:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ImportError as exc:
            raise RuntimeError("MolmoPoint requires torch>=2.6 and transformers>=4.57.1 in the optional environment") from exc
        self._torch = torch
        self.model_name = model_name or os.environ.get("CODEX_MOLMOPOINT_MODEL") or "allenai/MolmoPoint-8B"
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._processor = AutoProcessor.from_pretrained(
            self.model_name, trust_remote_code=True, padding_side="left")
        self._model = AutoModelForImageTextToText.from_pretrained(
            self.model_name, trust_remote_code=True,
            dtype=torch.bfloat16 if self._device == "cuda" else torch.float32,
            device_map=self._device,
        )
        if not callable(getattr(self._model, "extract_image_points", None)):
            raise RuntimeError("MolmoPoint checkpoint does not expose native extract_image_points")

    @staticmethod
    def _point_from_rows(rows, size):
        if not rows:
            return None
        if len(rows) != 1:
            raise ValueError("MolmoPoint returned multiple native points")
        row = rows[0]
        if len(row) < 2:
            raise ValueError("MolmoPoint native point row is incomplete")
        width, height = size
        xy = []
        for value, bound in zip(row[-2:], (width, height)):
            if isinstance(value, bool):
                raise ValueError("MolmoPoint returned a nonnumeric point")
            try:
                value = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError("MolmoPoint returned a nonnumeric point") from exc
            if not math.isfinite(value) or not 0 <= value <= bound:
                raise ValueError("MolmoPoint returned a point outside the image")
            xy.append(value * 100 / bound)
        return xy

    def _query(self, image, prompt):
        torch = self._torch
        messages = [{"role": "user", "content": [
            {"type": "text", "text": prompt}, {"type": "image", "image": image},
        ]}]
        inputs = self._processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt",
            return_dict=True, padding=True, return_pointing_metadata=True)
        metadata_ = inputs.pop("metadata")
        inputs = {key: value.to(self._device) for key, value in inputs.items()}
        autocast = (torch.autocast("cuda", dtype=torch.bfloat16)
                    if self._device == "cuda" else nullcontext())
        with torch.inference_mode(), autocast:
            output = self._model.generate(
                **inputs,
                logits_processor=self._model.build_logit_processor_from_inputs(inputs),
                max_new_tokens=200, do_sample=False,
            )
        tokens = output[:, inputs["input_ids"].size(1):]
        raw = self._processor.post_process_image_text_to_text(
            tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False)[0]
        rows = self._model.extract_image_points(
            raw, metadata_["token_pooling"], metadata_["subpatch_mapping"],
            metadata_["image_sizes"])
        return raw, rows

    def predict(self, image: Image.Image, task: str, object_name: str,
                descriptions: dict[str, str]) -> dict:
        _validate_request(image, task, object_name, descriptions)
        image = image.convert("RGB")
        result = {}
        for kind, description in descriptions.items():
            prompt = _prompt(task, object_name, {kind: description}) + (
                "\nUse native image pointing for the one visible point. "
                "If it is not visible or identifiable, return no point."
            )
            raw, rows = self._query(image, prompt)
            point = self._point_from_rows(rows, image.size)
            result[kind] = {
                "visible": point is not None, "point": point,
                "description": description, "raw": raw, "model": getattr(self, "model_name", "MolmoPoint"),
            }
        return result


def create_point_backend(name="sol"):
    """Create a lazy-loaded Sol or MolmoPoint backend."""
    if name == "sol":
        return SolPointing()
    if name == "molmopoint":
        return MolmoPointPointing()
    raise ValueError(f"unknown point backend: {name!r}; choose 'sol' or 'molmopoint'")
