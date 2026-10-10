#!/bin/bash

# CoDex installation script.
#
# Installs the codex package, its Python dependencies, LangSAM, and (by
# default) FoundationPose into a conda environment.
#
# Usage:
#   bash install.sh --env-name codex                    # full install
#   bash install.sh --env-name codex --no-foundationpose  # skip pose estimation
#
# Options:
#   --env-name <name>        Conda env to create/use (recommended).
#   --no-foundationpose      Skip FoundationPose (mesh-in-scene pose estimation,
#                            needed to create new scenes; bundled example scenes work without it).
#   --python-version <ver>   Python version for a new env (default 3.11.11).

set -e

# Fail loudly with context instead of dying silently.
trap 'echo ""; echo "[install.sh] FAILED at line $LINENO while running:"; echo "    $BASH_COMMAND"; echo "Fix the issue above and re-run install.sh (completed steps are skipped or safely re-run)."' ERR

# Versions known to work together (CUDA 11.8 toolchain).
TORCH_VERSION="2.5.0"
TORCHVISION_VERSION="0.20.0"
TORCH_INDEX_URL="https://download.pytorch.org/whl/cu118"
CUDA_LABEL="nvidia/label/cuda-11.8.0"
LANGSAM_COMMIT="19cb3670caea21e11b35e01ce1115a4a35b27fb5"
NVDIFFRAST_COMMIT="253ac4fcea7de5f396371124af597e6cc957bfae"
PYTORCH3D_TAG="V0.7.8"
FOUNDATIONPOSE_BASE_COMMIT="b41bc340a4c2346f4bbc47f9aa77e48fb04107fc"
FOUNDATIONPOSE_PATCHED_COMMIT="cf73e97daf93608c0ea74674a48f2e652a2951b3"

INSTALL_FOUNDATIONPOSE=true
ENV_NAME=""
PYTHON_VERSION="3.11.11"

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --install-foundationpose) INSTALL_FOUNDATIONPOSE=true; shift ;;
        --no-foundationpose) INSTALL_FOUNDATIONPOSE=false; shift ;;
        --env-name|--python-version)
            if [ "$#" -lt 2 ] || [ -z "$2" ] || [[ "$2" == --* ]]; then
                echo "Missing value for $1"
                exit 2
            fi
            if [ "$1" = --env-name ]; then ENV_NAME="$2"; else PYTHON_VERSION="$2"; fi
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: bash install.sh [--env-name <name>] [--no-foundationpose] [--python-version <version>]"
            exit 1
            ;;
    esac
done

if [ "$INSTALL_FOUNDATIONPOSE" = false ]; then
    echo "NOTE: skipping FoundationPose (--no-foundationpose)."
    echo "Creating NEW scenes (image or camera capture) will be unavailable:"
    echo "stage 0 uses FoundationPose to register the object mesh into the scene."
    echo "The bundled example scenes include a cached pose and work without it."
fi

if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda not found on PATH. Install Miniconda/Anaconda first:"
    echo "    https://docs.conda.io/en/latest/miniconda.html"
    echo "then re-run this script from a shell where 'conda' works."
    exit 1
fi

if [ -n "$ENV_NAME" ]; then
    eval "$(conda shell.bash hook)"
    if conda info --envs | awk -v target="$ENV_NAME" '$1 == target { found = 1 } END { exit !found }'; then
        # `read` fails at EOF; default to reuse so non-interactive re-runs work.
        read -p "Conda environment '$ENV_NAME' already exists. Start fresh and overwrite it? (y/N): " choice || choice=""
        case "$choice" in
            y|Y )
                echo "Removing existing environment '$ENV_NAME'..."
                conda env remove -y -n "$ENV_NAME"
                conda create -y -n "$ENV_NAME" -c conda-forge --override-channels python="$PYTHON_VERSION"
                ;;
            * )
                echo "Reusing existing environment '$ENV_NAME'..."
                ;;
        esac
    else
        conda create -y -n "$ENV_NAME" -c conda-forge --override-channels python="$PYTHON_VERSION"
    fi
    conda activate "$ENV_NAME"
elif [ -z "$CONDA_DEFAULT_ENV" ] || [ "$CONDA_DEFAULT_ENV" = "base" ]; then
    echo "ERROR: no --env-name given and no conda environment is active (or 'base' is)."
    echo "Refusing to install into base/system Python. Run:"
    echo "    bash install.sh --env-name codex"
    exit 1
else
    echo "Installing into the active conda environment: $CONDA_DEFAULT_ENV"
fi

# Resolve project root (this script lives at the repo root).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ ! -f "$SCRIPT_DIR/setup.py" ]; then
    echo "Could not locate project root (expected setup.py next to install.sh)."
    exit 1
fi
PROJECT_ROOT="$SCRIPT_DIR"
cd "$PROJECT_ROOT"

echo "========================================"
echo "CoDex installation"
echo "  Project root:        $PROJECT_ROOT"
echo "  Conda env:           ${ENV_NAME:-$CONDA_DEFAULT_ENV}"
echo "  Torch:               $TORCH_VERSION (cu118)"
echo "  FoundationPose:      $INSTALL_FOUNDATIONPOSE"
echo "========================================"

echo ""
echo "[1/6] pip + setuptools"
python -m pip install --upgrade pip
# FoundationPose's CUDA extensions need setuptools < 82 to build.
python -m pip install "setuptools==75.6.0"

echo ""
echo "[2/6] PyTorch $TORCH_VERSION (CUDA 11.8)"
# Installed first and pinned so every later step (pytorch-kinematics,
# nvdiffrast, LangSAM, ...) builds/runs against the same torch.
python -m pip install "torch==$TORCH_VERSION" "torchvision==$TORCHVISION_VERSION" --index-url "$TORCH_INDEX_URL"

echo ""
echo "[3/6] CoDex Python dependencies + package"
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps

echo ""
echo "[4/6] LangSAM (language-conditioned segmentation)"
mkdir -p third_parties
if [ ! -d "third_parties/lang-segment-anything" ]; then
    git clone https://github.com/luca-medeiros/lang-segment-anything.git third_parties/lang-segment-anything
fi
git -C third_parties/lang-segment-anything checkout -q "$LANGSAM_COMMIT"
# LangSAM pins exact torch/torchvision versions; relax them so it uses the
# torch installed above instead of downgrading it. Its gradio pin (for its own
# demo UI, unused here) is relaxed too, to allow the app's gradio 5.12.
sed -i.bak 's/torch==/torch>=/g; s/torchvision==/torchvision>=/g; s/torchaudio==/torchaudio>=/g; s/gradio==[0-9.]*/gradio>=5.0.2/g' third_parties/lang-segment-anything/requirements.txt
python -m pip install -e third_parties/lang-segment-anything

if [ "$INSTALL_FOUNDATIONPOSE" = true ]; then
    echo ""
    echo "[5/6] FoundationPose (pose estimation; builds CUDA extensions)"
    # Build toolchain for the CUDA extensions (nvdiffrast, mycuda, pysdf).
    conda install -y -c conda-forge gcc=11 gxx=11 libstdcxx-ng eigen=3.4.0 boost-cpp cmake make
    conda install -y -c "$CUDA_LABEL" cuda-toolkit
    export CMAKE_PREFIX_PATH="$CMAKE_PREFIX_PATH:$CONDA_PREFIX/include/eigen3"
    # Make the env's libstdc++ visible at runtime (nvdiffrast needs GLIBCXX_3.4.29),
    # persistently for the env, and for the remainder of this script.
    conda env config vars set "LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH" > /dev/null
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

    if [ ! -d "third_parties/FoundationPose" ]; then
        git clone https://github.com/JiangBowen0008/FoundationPose.git third_parties/FoundationPose
        git -C third_parties/FoundationPose checkout --detach "$FOUNDATIONPOSE_BASE_COMMIT"
    fi
    # The fork's two validated memory fixes are not on its public branch.
    # Apply the checked-in diff to the pinned public commit. An existing local
    # checkout at the already-patched commit, or a rerun with the diff applied,
    # is accepted without changing that checkout.
    bash "$PROJECT_ROOT/scripts/apply_foundationpose_patch.sh" \
        "$PROJECT_ROOT/third_parties/FoundationPose" \
        "$FOUNDATIONPOSE_BASE_COMMIT" "$FOUNDATIONPOSE_PATCHED_COMMIT" \
        "$PROJECT_ROOT/patches/foundationpose/b41bc34-to-cf73e97.patch"
    cd third_parties/FoundationPose
    # pybind11 must be present before requirements.txt so pysdf can build.
    python -m pip install pybind11
    python -m pip install -r requirements.txt --no-build-isolation
    python -m pip install --no-cache-dir --no-build-isolation "git+https://github.com/NVlabs/nvdiffrast.git@$NVDIFFRAST_COMMIT"
    python -m pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@$PYTORCH3D_TAG"
    python -m pip install -e .

    # Build the mycpp extension (pose-grid clustering; CMake project, not
    # covered by the pip install).
    (cd foundationpose/mycpp && rm -rf build && mkdir -p build && cd build && \
        cmake .. -DPYTHON_EXECUTABLE="$(which python)" \
                 -Dpybind11_DIR="$(python -m pybind11 --cmakedir)" \
                 -DCMAKE_PREFIX_PATH="$CONDA_PREFIX" && \
        make -j"${CMAKE_BUILD_PARALLEL_LEVEL:-4}")

    # Network weights (refiner 2023-10-28-18-33-37 + scorer 2024-01-11-20-02-45).
    if [ ! -d "foundationpose/weights/2024-01-11-20-02-45" ]; then
        python -m pip install gdown
        python -m gdown --folder 1DFezOAD0oD1BblsXVxqDsl8fj0qzB82i -O foundationpose/weights || {
            echo "WARNING: automatic weights download failed. Download them manually"
            echo "(link in third_parties/FoundationPose/readme.md) into"
            echo "third_parties/FoundationPose/foundationpose/weights/"
        }
    fi
    cd "$PROJECT_ROOT"
    # FoundationPose's requirements are broad and bump a few packages the rest
    # of the stack pins; restore the validated versions. Its old Rtree and
    # ninja wheels have malformed WHEEL metadata that fails `pip check` on
    # current pip; these newer wheels keep the native modules working.
    python -m pip install "pydantic==2.9.2" "wandb==0.16.5" "websockets>=13,<17" "fastapi==0.115.11" "starlette==0.46.0" "uvicorn==0.34.0" "rtree==1.4.1" "ninja==1.13.2"
else
    echo ""
    echo "[5/6] FoundationPose: skipped"
fi

echo ""
echo "[6/6] Verifying installation"
python -m pip check
python - <<'PYEOF'
import importlib, sys

checks = [
    ("codex", "codex package"),
    ("torch", "PyTorch"),
    ("pytorch_kinematics", "pytorch-kinematics (grasp solver)"),
    ("open3d", "Open3D"),
    ("trimesh", "trimesh"),
    ("mani_skill", "ManiSkill3 (RL training)"),
    ("stable_baselines3", "Stable-Baselines3"),
    ("lang_sam", "LangSAM (segmentation)"),
    ("gradio", "Gradio (step-through app)"),
]
failed = []
for module, label in checks:
    try:
        importlib.import_module(module)
        print(f"  ok      {label}")
    except Exception as e:
        print(f"  FAILED  {label}: {type(e).__name__}: {e}")
        failed.append(label)

import torch
if torch.cuda.is_available():
    print(f"  ok      CUDA available ({torch.cuda.get_device_name(0)})")
else:
    print("  WARNING CUDA not available - grasp synthesis and RL training need a GPU.")

sys.exit(1 if failed else 0)
PYEOF

if [ "$INSTALL_FOUNDATIONPOSE" = true ]; then
    python -c "
from foundationpose import Utils
assert Utils.mycpp is not None, 'mycpp extension not built'
import os
w = os.path.join('third_parties/FoundationPose/foundationpose/weights', '2024-01-11-20-02-45')
assert os.path.isdir(w), 'weights missing: ' + w
" \
        && echo "  ok      FoundationPose (incl. mycpp + weights)" \
        || { echo "  FAILED  FoundationPose (import/mycpp/weights)"; exit 1; }
fi

echo ""
echo "========================================"
echo "Installation complete."
echo "Next steps: set your API keys and run the pipeline (see README.MD):"
echo "    export OPENAI_API_KEY=...   TRIPO_API_KEY=..."
echo "========================================"
