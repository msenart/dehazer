# Dehazer

**Dehazer** is a single-image haze removal tool built around the Dark Channel Prior algorithm (He et al., 2009), with a PySide6 desktop GUI for running, tuning, and comparing dehazing pipelines.

## How it works

The pipeline follows the classic Dark Channel Prior approach:

1. **Dark channel** — for each pixel, take the minimum over color channels, then erode over a local window. Haze-free outdoor images have a dark channel close to zero almost everywhere.
2. **Atmospheric light (A)** — estimated from the brightest pixels among those with the highest dark-channel values (assumed to be haze-only regions).
3. **Coarse transmission map** — derived from the dark channel of the image normalized by `A`.
4. **Transmission refinement** — the coarse map is blocky, so it's refined against the original image using one of three interchangeable methods:
   - **Guided filter** ([u_guided_filter.py](dehazer/u_guided_filter.py)) — fast, edge-aware smoothing.
   - **Soft matting** ([u_soft_matting.py](dehazer/u_soft_matting.py)) — closed-form matting Laplacian solved with conjugate gradient; higher quality, slower, parallelized across CPU cores.
   - **Chunked soft matting** ([u_soft_matting_chunked.py](dehazer/u_soft_matting_chunked.py)) — the same soft matting method applied tile-by-tile so it scales to larger images.
5. **Radiance recovery** — the haze-free image is recovered from the original image, `A`, and the refined transmission map.

This is implemented in [dehazer/core.py](dehazer/core.py).

## Project structure

The project is a Python package, [dehazer/](dehazer/):

- `__main__.py` — entry point for `python -m dehazer` (launches the GUI).
- `gui.py` — PySide6 GUI application (drag-and-drop dehazing pipeline, processing queue, image viewer, and comparator).
- `core.py` — Core Dark Channel Prior dehazing algorithm (dark channel, atmospheric light, transmission estimation, radiance recovery).
- `u_guided_filter.py`, `u_soft_matting.py`, `u_soft_matting_chunked.py` — Transmission-map refinement methods.
- `image_diff.py` — Utility to compare two images channel by channel.
- `config.py` — Shared project paths (`PROJECT_ROOT`, `OUTPUT_DIR`).
- `comparison.py`, `plot_comparison.py`, `psnrssim.py` — Batch evaluation scripts (MSE/PSNR/SSIM/UIQM metrics) and plotting utilities for comparing dehazed output against ground truth.

The [Final report/](Final%20report/) folder contains the project's final report and slides (in French) as well as the mid-term report.

## Getting started

Install the dependencies:

```
pip install -r requirements.txt
```

Then launch the GUI from the project root:

```
python -m dehazer
```

This opens a desktop interface where you can:

- Drop an image and choose which dehazing method to apply
- Configure and run the processing pipeline
- Compare two images or two full pipelines to visualize differences between methods

Processed images and their `params.json` metadata are written to `seriespicturesoutput/` at the project root (git-ignored).

## Evaluation scripts

The batch evaluation utilities are modules within the package, so run them the same way, from the project root:

```
python -m dehazer.comparison --gt <ground_truth.png> --results_root <folder>
python -m dehazer.plot_comparison --results_root <folder>
python -m dehazer.psnrssim
```

`comparison.py` and `plot_comparison.py` accept `--help` for their full list of options; `psnrssim.py` sweeps the dark-channel-size parameter over the I-HAZE dataset and its inputs are configured as constants at the top of the file.

## Test images

To try the project out, you can download the following haze datasets:

- I-Haze: artificial indoor hazy images — **http://www.vision.ee.ethz.ch/ntire18/i-haze/I-HAZE.zip**
- O-Haze: artificial outdoor hazy images — **http://www.vision.ee.ethz.ch/ntire18/o-haze/O-HAZE.zip**
- NH-Haze: non-homogeneous hazy images — **https://data.vision.ee.ethz.ch/cvl/ntire20/nh-haze/files/NH-HAZE.zip**
- D-Haze: depth-dependent hazy images — **http://ancuti.meo.etc.upt.ro/D_Hazzy_ICIP2016/D-HAZY_DATASET.zip**

Download a dataset with `curl <link>` and extract it into a `hazed_images/` folder at the project root (this folder is git-ignored) to use it with the evaluation scripts above.

## License

Apache License 2.0 — see [LICENSE](LICENSE).
