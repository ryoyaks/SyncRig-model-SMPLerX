# syncrig-model-smplerx

SMPLer-X provider plug-in for [SyncRig](https://github.com/ryoyaks/SyncRig).
Adds whole-body SMPL-X estimation (22 body + jaw + 2 eyes + 30 fingers
= 55 kinematic joints) plus a 10,475-vertex SMPL-X mesh from a single
RGB frame.

## ⚠ Non-commercial use only

This package ships **research-only** weights and depends on a
**research-only** parametric model. Commercial deployment requires
separate licenses; see the licensing section below before installing.

This is why the package is its own repo rather than being bundled with
the public SyncRig — keeping the license footprint of mainline
SyncRig clean (Apache 2.0 / BSD / MIT) lets commercial users adopt it
without inheriting research-only constraints.

## Install

```bash
# 1. Get the package + its torch / smplx runtime deps:
uv pip install 'syncrig-model-smplerx[runtime]'
# (or)  pip install 'syncrig-model-smplerx[runtime]'

# 2. Fetch the SMPLer-X checkpoint:
uv pip install huggingface-hub
hf download caizhongang/SMPLer-X smpler_x_s32.pth.tar \
    --local-dir ./models/smplerx

# 3. Provide the SMPL-X parametric model (license-gated):
#    https://smpl-x.is.tue.mpg.de/  →  models_smplx_v1_1.zip
#    → extract SMPLX_NEUTRAL.npz to ./models/smplerx/
```

The engine resolves `models/` relative to its repo root by default, or
to `$SYNCRIG_MODELS_DIR` if you set that env var.

## How it plugs in

The package exposes one [`syncrig.providers`][ep] entry-point:

```toml
[project.entry-points."syncrig.providers"]
smplerx = "syncrig_model_smplerx.provider:SmplerXProvider"
```

On engine startup, SyncRig calls
`ProviderRegistry.autoload_entry_points("syncrig.providers")` which
finds this entry-point, imports the package (triggering the topology +
retarget handler registration in `syncrig_model_smplerx/__init__.py`)
and registers the provider class. The Extensions page card shows up
automatically with the engine-supplied description + non-commercial
chip.

No fork of SyncRig core required.

[ep]: https://packaging.python.org/en/latest/specifications/entry-points/

## Licensing

The package itself is **S-Lab License 1.0** (see `LICENSE`). At runtime
it pulls in additional license-restricted dependencies:

| Layer | License | Commercial? |
|---|---|---|
| SMPLer-X code (this repo) | S-Lab 1.0 | ❌ Research only |
| SMPLer-X checkpoints | S-Lab 1.0 | ❌ Research only |
| `smplx` PyPI package | MPI research license | ❌ Research only |
| `SMPLX_NEUTRAL.npz` | MPI SMPL-X research license | ❌ Research only |

For commercial deployment you need separate licenses from **S-Lab**
(for code + weights) and **Meshcapade** (for the SMPL-X parametric
model). Mocap output remains under the same restrictions.

## Cite

If you use this in research:

```bibtex
@inproceedings{cai2023smplerx,
  title     = {{SMPLer-X}: Scaling Up Expressive Human Pose and Shape Estimation},
  author    = {Cai, Zhongang and Yin, Wanqi and Zeng, Ailing and Wei, Chen and Sun, Qingping and Wang, Yanjun and Pang, Hui En and Mei, Haiyi and Zhang, Mingyuan and Zhang, Lei and Loy, Chen Change and Yang, Lei and Liu, Ziwei},
  booktitle = {NeurIPS},
  year      = {2023},
}
```
