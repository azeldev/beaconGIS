# beaconGIS — Building Damage Assessment for QGIS

AI-powered building-level damage classification from pre-/post-disaster RGB satellite or aerial imagery, integrated directly into QGIS.

The plugin uses a Siamese U-Net ensemble (SeResNeXt-50 32x4d encoder) trained on the full public xView2 / xBD dataset and runs entirely on ONNX Runtime — **no PyTorch install required**. Designed as a draft-generation tool for disaster-response analysts: output should be reviewed by humans, not used as authoritative ground truth.

> 📊 **Combined F1 = 0.7312** (xView2 official scorer, full plugin pipeline)
> 📦 **Plugin install footprint**: ~150 MB of Python deps + ~234 MB of AI weights (downloaded on first run, fp16)

## Features

- **End-to-end damage classification** — input is a pair of RGB rasters, output is a categorized polygon vector layer (No Damage / Minor / Major / Destroyed) with per-polygon confidence.
- **Tile-based inference** with optional 4-way test-time augmentation and seamless Hanning-window blending for arbitrarily large rasters.
- **Two-model ensemble** — independently-trained classification models are softmax-averaged for a more robust per-pixel prediction than any single model produces.
- **CPU Fast Mode** for non-GPU machines — drops TTA and ensemble for a ~8× speedup at ~0.5 F1-point cost.
- **GPU support without CUDA Toolkit** — install `onnxruntime-directml` (any DX12 GPU on Windows) or `onnxruntime-gpu` (NVIDIA CUDA). The plugin auto-detects and picks the fastest available provider.
- **GSD normalization** — automatically resamples non-xBD imagery to the model's training resolution (~0.5 m/px), preserving georeferencing.
- **Automatic pre/post co-registration** — global phase-correlation + ECC translation alignment plus bounded per-tile refinement, directly attacking the misregistration/parallax failure mode on non-xBD pairs. CRS/extent mismatches are detected geographically and warped, even when the two rasters have identical pixel dimensions.
- **Background execution** — detection (and the first-run weights download) runs as a QGIS task: the UI stays responsive, with live progress and Cancel in the task manager.
- **Processing integration** — `beacongis:detectbuildingdamage` in the Processing Toolbox enables batch runs, headless `qgis_process` execution, and Model Designer chaining.
- **Calibrated confidence (optional)** — ship a `calibration.json` (fit with `calibrate_temperature.py` on a held-out xBD split) and per-building confidence becomes temperature-scaled, making confidence-threshold triage statistically meaningful.
- **AOI clipping**, GeoPackage export, per-pixel damage mask GeoTIFF export, and JSON metadata sidecars for reproducible outputs.

## Requirements

- **QGIS 3.16 or later** (tested through 3.34 / 3.40).
- Python packages installed in QGIS's Python environment:

```bash
python -m pip install onnxruntime numpy pillow scipy opencv-python
```

**GPU acceleration (optional, recommended)** — pick ONE:

```bash
# Windows with any DirectX 12 GPU (Intel/AMD/NVIDIA — no CUDA Toolkit needed):
python -m pip install onnxruntime-directml

# NVIDIA CUDA (requires matching CUDA runtime):
python -m pip install onnxruntime-gpu
```

On the OSGeo4W shell (Windows) replace `python` with `python-qgis` to install into the QGIS-bundled Python.

## Installation

### Option A — QGIS Plugin Manager (recommended, once published)

1. *Plugins → Manage and Install Plugins → All*
2. Search for `beaconGIS`
3. Click **Install Plugin**
4. Restart QGIS

### Option B — Manual installation

1. Download or clone this repository.
2. Copy the `change_detector/` folder to your QGIS plugins directory:
   - **Windows**: `C:\Users\<you>\AppData\Roaming\QGIS\QGIS3\profiles\default\python\plugins\`
   - **macOS**: `~/Library/Application Support/QGIS/QGIS3/profiles/default/python/plugins/`
   - **Linux**: `~/.local/share/QGIS/QGIS3/profiles/default/python/plugins/`
3. Restart QGIS and enable the plugin: *Plugins → Manage and Install Plugins → Installed → ✅ beaconGIS*.

## First-run setup

The plugin downloads ~234 MB of fp16 model weights from the project's [GitHub releases page](https://github.com/azeldev03/beaconGIS/releases) on first run. No account, no token, no form — plain HTTPS download, SHA-256 verified end-to-end. Subsequent runs use the local cache.

Manual install: download the three `.onnx` files from the [latest release](https://github.com/azeldev03/beaconGIS/releases/latest) and drop them into the plugin folder (e.g. `~/AppData/Roaming/QGIS/QGIS3/profiles/default/python/plugins/change_detector/`).

The model weights themselves are released under **CC BY-NC-SA 4.0** (academic / humanitarian use only; see [LICENSE](LICENSE)). The plugin code is GPL v2+ to match QGIS conventions.

## Usage

1. **Load** pre-disaster and post-disaster RGB rasters into QGIS as raster layers (GeoTIFF works best, but the plugin auto-handles CRS and resolution mismatches).
2. **Open the plugin** from the toolbar icon or *Plugins → beaconGIS → Detect Building Damage*.
3. **Select** the before and after layers from the dropdowns.
4. *(Optional)* **Select an AOI polygon layer** to restrict output to a specific area.
5. *(Optional)* **Toggle CPU Fast Mode** if you are on a laptop without a GPU.
6. *(Optional)* **Toggle TTA** for a small accuracy bump on production runs (slower).
7. *(Optional)* **Tune the Building Separation slider** if touching buildings merge or single buildings over-split.
8. *(Optional)* Tick **Save as GeoPackage** / **Save as GeoTIFF mask** if you want file outputs alongside the in-QGIS layer.
9. Click **OK**.

When detection completes, the **Assessment Reports panel** opens on the right with summary statistics, three offline template-driven report styles (Damage Pattern Analysis / Full SitRep / Response Priorities), and CSV / PDF / HTML / TXT export. All report generation is local — no LLM API key required.

## Model details

| Component | Details |
|-----------|---------|
| Localization | LocalizationUNet, SeResNeXt-50 (32x4d) encoder |
| Classification | SiameseUNet, SeResNeXt-50 (32x4d) encoder, 4-class output |
| Ensemble | cls_best.onnx + cls_m2.onnx (independent seeds, fused via softmax-mean) |
| Training data | ImageNet → Inria Aerial → xBD (Tier 1 + Tier 3 + earthquake-augmented sampling) |
| Combined F1 (xView2 scorer) | **0.7312** |
| Per-class F1 (cls_best) | NoDmg 0.918, Minor 0.437, Major 0.750, Destroyed 0.825 |
| Per-class F1 (cls_m2) | NoDmg 0.925, Minor 0.497, Major 0.656, Destroyed 0.828 |
| Distribution format | ONNX (opset 17, dynamic axes on batch/H/W) |
| Plugin runtime | ONNX Runtime ≥ 1.17 + numpy + scipy + Pillow + OpenCV |

## Limitations (honest)

- **xBD-style imagery** (Maxar pre + Maxar post, well-aligned, near-nadir): works well — this is what the model was trained on.
- **Same-sensor cross-event imagery** (e.g. Maxar Open Data captures of recent disasters): works as a draft. Expect ~70 % useful polygons that an analyst can refine in 15 minutes.
- **Cross-sensor or off-nadir pairs** (e.g. archive Maxar pre + post-event Pléiades, or drone surveys): partially handled by GSD normalization, but residual artifacts remain. Off-nadir scenes systematically over-predict damage from rooftop parallax — review against the source imagery before publishing.
- **Minor damage class is the hardest** (F1 ≈ 0.44–0.50). Filter the damage layer by `damage_class = 2` and the `confidence` attribute to triage uncertain buildings before publishing.
- **The plugin is a draft generator, not an authoritative result.** Always review output before using it for response, allocation, or policy decisions.

## Privacy

- **Imagery, predictions, and reports stay on your machine.** The plugin does not transmit your input rasters, model outputs, or generated reports to any external service.
- **First-run weights download** is a plain HTTPS GET against this plugin's GitHub release page. No authentication, no token, no telemetry. This is the only outbound traffic the damage-detection feature generates.
- **Assessment Reports** are generated entirely locally from your stats — no LLM API, no telemetry, no key configuration.
- **Optional Satellite Downloader** (toolbar button) fetches public tile-server imagery (ESRI World Imagery / OpenStreetMap) via HTTPS.

## License

- **Plugin code**: GPL v2+ (see [LICENSE](LICENSE)) — matches the QGIS host application's license.
- **Model weights**: CC BY-NC-SA 4.0 — non-commercial use only; share-alike on derivatives. The weights' license is required by the upstream xBD dataset terms and carries forward.

## Citation

If you use this plugin in research or response work, please cite both:

```bibtex
@misc{ozel2026beacongis,
  author       = {Özel, Adem},
  title        = {beaconGIS: AI Building Damage Detection for QGIS},
  year         = {2026},
  howpublished = {QGIS plugin},
  url          = {https://github.com/azeldev03/beaconGIS},
}

@article{gupta2019xbd,
  title   = {{xBD}: A Dataset for Assessing Building Damage from Satellite Imagery},
  author  = {Gupta, Ritwik and Goodman, Bryce and Patel, Nirav and others},
  journal = {arXiv preprint arXiv:1911.09296},
  year    = {2019},
}
```

## Acknowledgments

- **xView2 / xBD** dataset (Gupta et al. 2019)
- **Inria Aerial Image Labeling** dataset
- **vdurnov's xView2 first-place solution** (2019) — architectural inspiration
- **timm** (Ross Wightman) — the SeResNeXt-50 encoder
- **ONNX Runtime** (Microsoft) — the inference backend that lets this plugin ship without a 1.5 GB PyTorch install
- **The Deepness QGIS plugin** (Aszkowski et al. 2023, *SoftwareX*) — packaging conventions

## Contact

**Adem Özel** — <azel.dev03@gmail.com>
Issues & feature requests: <https://github.com/azeldev03/beaconGIS/issues>
