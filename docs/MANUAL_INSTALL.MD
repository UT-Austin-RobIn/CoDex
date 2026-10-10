# Manual Installation Guide

Use this guide only if you cannot run `install.sh` or need to customize the
install sequence. Otherwise, prefer:

```bash
bash install.sh --env-name codex --no-foundationpose
```

Install from a full source checkout at the repository root. The project relies
on repository assets and an editable package install; a standalone wheel or
PyPI install is not supported.

Use Linux, Python 3.11.11 and an NVIDIA driver supporting CUDA 11.8.
The steps below follow the installer's main sequence.

## 1. Create and Activate a Conda Environment

```bash
conda create -y -n codex -c conda-forge --override-channels python=3.11.11   # no Anaconda ToS prompt
conda activate codex
python -m pip install --upgrade pip
python -m pip install setuptools==75.6.0   # FoundationPose CUDA builds need setuptools < 82
```

## 2. Install PyTorch (first, pinned)

Everything later builds/runs against this torch, so install it before
anything else:

```bash
python -m pip install torch==2.5.0 torchvision==0.20.0 --index-url https://download.pytorch.org/whl/cu118
```

## 3. Install CoDex Dependencies and Package

```bash
# From the project root
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

`requirements.txt` pins `dash<4` because Dash 4 requires a newer Pydantic than
the pinned LangSAM stack supports.

## 4. Install LangSAM (segmentation)

LangSAM pins exact torch versions; relax them so it reuses the torch from
step 2 instead of downgrading it:

```bash
mkdir -p third_parties
git clone https://github.com/luca-medeiros/lang-segment-anything.git third_parties/lang-segment-anything
git -C third_parties/lang-segment-anything checkout 19cb3670caea21e11b35e01ce1115a4a35b27fb5
sed -i 's/torch==/torch>=/g; s/torchvision==/torchvision>=/g; s/torchaudio==/torchaudio>=/g; s/gradio==[0-9.]*/gradio>=5.0.2/g' \
    third_parties/lang-segment-anything/requirements.txt
python -m pip install -e third_parties/lang-segment-anything
```

## 5. Install FoundationPose (optional — mesh-in-scene pose estimation)

Needed to create new scenes (stage 0); the bundled example scenes include a
cached `pose.txt` and run without it.
See [FoundationPose Setup Commands](FOUNDATIONPOSE_SETUP_COMMANDS.MD) for the
step-by-step commands (build toolchain, CUDA toolkit, nvdiffrast, pytorch3d,
and the FoundationPose package itself).

## 6. Verify

```bash
python -m pip check
python -c "import codex, torch, pytorch_kinematics, lang_sam, mani_skill; print('ok, cuda:', torch.cuda.is_available())"
```
