# FoundationPose Setup Commands

These are the commands `install.sh` runs for its FoundationPose section
(step 5), for use when installing manually. Run them from the project root
with your conda environment active, **after** installing torch and the main
requirements (steps 1–4 of the [Manual Installation Guide](MANUAL_INSTALL.MD)).

FoundationPose builds CUDA extensions (nvdiffrast, pysdf, mycuda), which is
why it needs a compiler toolchain and the CUDA 11.8 toolkit inside the env.

## 1. Build toolchain + CUDA toolkit

```bash
conda install -y -c conda-forge gcc=11 gxx=11 libstdcxx-ng eigen=3.4.0 boost-cpp cmake make
conda install -y -c "nvidia/label/cuda-11.8.0" cuda-toolkit
export CMAKE_PREFIX_PATH="$CMAKE_PREFIX_PATH:$CONDA_PREFIX/include/eigen3"

# Make the env's libstdc++ visible at runtime (nvdiffrast needs GLIBCXX_3.4.29).
# `conda env config vars set` makes this persist for future activations.
conda env config vars set "LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
```

## 2. Clone and install FoundationPose

```bash
git clone https://github.com/JiangBowen0008/FoundationPose.git third_parties/FoundationPose
git -C third_parties/FoundationPose checkout --detach b41bc340a4c2346f4bbc47f9aa77e48fb04107fc
bash scripts/apply_foundationpose_patch.sh \
    third_parties/FoundationPose \
    b41bc340a4c2346f4bbc47f9aa77e48fb04107fc \
    cf73e97daf93608c0ea74674a48f2e652a2951b3 \
    "$PWD/patches/foundationpose/b41bc34-to-cf73e97.patch"
cd third_parties/FoundationPose

# pybind11 must be installed before requirements.txt so pysdf can build.
python -m pip install pybind11
python -m pip install -r requirements.txt --no-build-isolation

python -m pip install --no-cache-dir --no-build-isolation \
    "git+https://github.com/NVlabs/nvdiffrast.git@253ac4fcea7de5f396371124af597e6cc957bfae"
python -m pip install --no-build-isolation \
    "git+https://github.com/facebookresearch/pytorch3d.git@V0.7.8"

python -m pip install -e .
cd ../..
```

## 3. Build the mycpp extension

Pose-grid clustering is a CMake project not covered by the pip install:

```bash
cd third_parties/FoundationPose/foundationpose/mycpp
mkdir -p build && cd build
cmake .. -DPYTHON_EXECUTABLE="$(which python)" \
         -Dpybind11_DIR="$(python -m pybind11 --cmakedir)" \
         -DCMAKE_PREFIX_PATH="$CONDA_PREFIX"
make -j"${CMAKE_BUILD_PARALLEL_LEVEL:-4}"
cd ../../../../..
```

The build uses four parallel jobs by default. Set `CMAKE_BUILD_PARALLEL_LEVEL`
before running `make` to choose a different job count.

## 4. Download network weights

Refiner `2023-10-28-18-33-37` and scorer `2024-01-11-20-02-45`, extracted
under `third_parties/FoundationPose/foundationpose/weights/` (link also in
the FoundationPose readme):

```bash
cd third_parties/FoundationPose
python -m pip install gdown
python -m gdown --folder 1DFezOAD0oD1BblsXVxqDsl8fj0qzB82i -O foundationpose/weights
cd ../..

# Restore versions compatible with the rest of the stack and replace the
# FoundationPose Rtree/ninja wheels whose WHEEL metadata fails `pip check`.
python -m pip install "pydantic==2.9.2" "wandb==0.16.5" "websockets>=13,<17" \
    "fastapi==0.115.11" "starlette==0.46.0" "uvicorn==0.34.0" \
    "rtree==1.4.1" "ninja==1.13.2"
```

## 5. Verify

```bash
python -c "
import foundationpose, nvdiffrast, pytorch3d
from foundationpose import Utils
assert Utils.mycpp is not None, 'mycpp extension not built'
print('ok')"
```
