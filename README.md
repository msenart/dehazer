# Dehazer

**Dehazer** is a single-image haze removal tool built around the Dark Channel Prior algorithm (He et al., 2009), with a PySide6 desktop GUI for running, tuning, and comparing dehazing pipelines.

## Project structure

The main code lives in [src/](src/):

- `main.py` — PySide6 GUI application (drag-and-drop dehazing pipeline, processing queue, image viewer, and comparator).
- `dehazer.py` — Core Dark Channel Prior dehazing algorithm (dark channel, atmospheric light, transmission estimation, radiance recovery).
- `u_guided_filter.py`, `u_soft_matting.py`, `u_soft_matting_chunked.py` — Transmission-map refinement methods (guided filter, closed-form soft matting, and a chunked/parallel variant for large images).
- `image_diff.py` — Utility to compare two images channel by channel.
- `comparison.py`, `plot_comparison.py`, `psnrssim.py` — Batch evaluation scripts (MSE/PSNR/SSIM/UIQM metrics) and plotting utilities for comparing dehazed output against ground truth.

The [Final report/](Final%20report/) folder contains the project's final report and slides (in French) as well as the mid-term report.

## Getting started

Install the dependencies:

```
pip install -r requirements.txt
```

Then launch the GUI from the `src/` folder:

```
python src/main.py
```

This opens a desktop interface where you can:

- Drop an image and choose which dehazing method to apply
- Configure and run the processing pipeline
- Compare two images or two full pipelines to visualize differences between methods

## Test images

To try the project out, you can download the following haze datasets:

- I-Haze: artificial indoor hazy images — **http://www.vision.ee.ethz.ch/ntire18/i-haze/I-HAZE.zip**
- O-Haze: artificial outdoor hazy images — **http://www.vision.ee.ethz.ch/ntire18/o-haze/O-HAZE.zip**
- NH-Haze: non-homogeneous hazy images — **https://data.vision.ee.ethz.ch/cvl/ntire20/nh-haze/files/NH-HAZE.zip**
- D-Haze: depth-dependent hazy images — **http://ancuti.meo.etc.upt.ro/D_Hazzy_ICIP2016/D-HAZY_DATASET.zip**

Download a dataset with `curl <link>` and extract it into a `hazed_images/` folder at the project root (this folder is git-ignored) to use it with the scripts in `src/`.
