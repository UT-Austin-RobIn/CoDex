"""Generate a 3D mesh from an image using the Tripo SDK.

Usage:
    python -m codex.mesh.image_to_mesh --scene-dir out/scenes/<scene_name> [--api-key KEY]

Uses obj_0.png (functional object) from the scene directory as input.
The script submits an image-to-model task, waits until completion, and downloads
the model files into <scene_dir>/mesh/.
"""

from __future__ import annotations

import argparse
import asyncio
import inspect
import sys
from pathlib import Path
from typing import Any

MAX_POLL_TIME = 600  # seconds



def _get_api_key(cli_key: str | None) -> str:
    from codex.utils.api_keys import get_key
    key = cli_key or get_key("TRIPO_API_KEY")
    if not key:
        print(
            "Error: Tripo API key not provided. "
            "Pass --api-key, set TRIPO_API_KEY, or save it in the app (API keys).",
            file=sys.stderr,
        )
        sys.exit(1)
    return key


def _filter_kwargs(func, kwargs: dict) -> dict:
    """Pass only kwargs accepted by func unless it supports **kwargs."""
    sig = inspect.signature(func)
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in sig.parameters.values()):
        return kwargs
    return {k: v for k, v in kwargs.items() if k in sig.parameters}


def _get_tripo_client_class() -> Any:
    try:
        from tripo3d import TripoClient  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "The 'tripo3d' package is required. Install it with: pip install tripo3d"
        ) from exc
    return TripoClient


async def _submit_image_task(client: Any, image_path: str) -> str:
    """Submit image-to-model task while handling minor SDK signature differences."""
    image_to_model = getattr(client, "image_to_model", None)
    if image_to_model is None:
        raise RuntimeError("Installed tripo3d package does not expose image_to_model().")

    attempts = [
        {"image": image_path},
        {"image_path": image_path},
        {"file_path": image_path},
        {"image_file": image_path},
    ]
    last_error: Exception | None = None
    for kwargs in attempts:
        try:
            return await image_to_model(**_filter_kwargs(image_to_model, kwargs))
        except TypeError as exc:
            last_error = exc
            continue
    raise RuntimeError(
        "Could not call TripoClient.image_to_model() with known argument names. "
        f"Last error: {last_error}"
    )


def _task_success(task) -> bool:
    status = getattr(task, "status", None)
    status_name = str(getattr(status, "name", status)).upper()
    return status_name == "SUCCESS"


def _pick_downloaded_glb(downloaded_files: dict[str, str], glb_path: Path) -> Path:
    for path_str in downloaded_files.values():
        candidate = Path(path_str)
        if candidate.suffix.lower() == ".glb":
            if candidate.resolve() != glb_path.resolve():
                glb_path.write_bytes(candidate.read_bytes())
                candidate.unlink(missing_ok=True)
            return glb_path

    first_path = next(iter(downloaded_files.values()), None)
    if first_path is None:
        raise RuntimeError("Tripo SDK did not return any downloaded files.")
    src = Path(first_path)
    if src.resolve() != glb_path.resolve():
        glb_path.write_bytes(src.read_bytes())
        src.unlink(missing_ok=True)
    return glb_path


async def _run_tripo_pipeline(api_key: str, image_path: str, out_dir: Path, glb_path: Path) -> None:
    TripoClient = _get_tripo_client_class()
    async with TripoClient(api_key=api_key) as client:
        task_id = await _submit_image_task(client, image_path)
        print(f"Created task {task_id}")

        wait_for_task_kwargs = _filter_kwargs(
            client.wait_for_task,
            {"task_id": task_id, "verbose": True, "timeout": MAX_POLL_TIME},
        )
        task = await client.wait_for_task(**wait_for_task_kwargs)
        if not _task_success(task):
            raise RuntimeError(f"Task {task_id} failed with status={getattr(task, 'status', None)}")

        download_kwargs = _filter_kwargs(
            client.download_task_models,
            {"task": task, "output_dir": str(out_dir), "directory": str(out_dir)},
        )
        downloaded_files = await client.download_task_models(**download_kwargs)
        saved_path = _pick_downloaded_glb(downloaded_files, glb_path)
        print(f"Saved mesh to {saved_path}")


def image_to_mesh(
    scene_dir: str,
    api_key: str | None = None,
) -> str:
    """End-to-end: scene_dir/obj_0.png -> Tripo API -> scene_dir/mesh/mesh.glb

    The downloaded mesh is raw (arbitrary scale/orientation); registration
    (MeshPoseEstimator.register_and_canonicalize) bakes its canonical frame.
    Returns the path to the output directory.
    """
    api_key = _get_api_key(api_key)

    scene_path = Path(scene_dir)
    image_path = str(scene_path / "obj_0.png")
    out_dir = scene_path / "mesh"
    out_dir.mkdir(parents=True, exist_ok=True)

    glb_path = out_dir / "mesh.glb"

    # Submit task, wait until success, and download outputs using SDK.
    asyncio.run(_run_tripo_pipeline(api_key, image_path, out_dir, glb_path))

    return str(out_dir)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a 3D mesh from a scene directory via the Tripo API."
    )
    parser.add_argument(
        "--scene-dir",
        dest="scene_dir",
        required=True,
        help="Scene directory containing obj_0.png and mesh/metadata.json. Mesh outputs are written to <scene_dir>/mesh/.",
    )
    parser.add_argument(
        "--api-key",
        dest="api_key",
        default=None,
        help="Tripo API key. Falls back to TRIPO_API_KEY env var.",
    )
    args = parser.parse_args()

    image_to_mesh(
        scene_dir=args.scene_dir,
        api_key=args.api_key,
    )


if __name__ == "__main__":
    main()
