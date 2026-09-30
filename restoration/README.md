# Image Restoration

`restore.py` applies four restoration options to images with one or more masked detector boxes:

1. Solid gray fill
2. OpenCV Telea inpainting
3. LaMa inpainting
4. Stable Diffusion inpainting

## Input Folders

Put images in folders under `restoration/data`. The script scans each immediate subfolder and supports `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tif`, and `.tiff` files. Images placed directly in `data` are also supported.

Each folder containing images must have a `detector_boxes.csv` file. The CSV maps image filenames to detector boxes in the 1280x736 detector-input coordinate system. Use one row per box; repeat the filename when an image has multiple boxes:

```csv
filename,x1,y1,x2,y2
image_a.jpg,100,120,180,200
image_a.jpg,400,220,480,300
image_b.jpg,520,250,600,330
```

The coordinates must satisfy `0 <= x1 < x2 <= 1280` and `0 <= y1 < y2 <= 736`. Every image needs at least one CSV row. The script combines all boxes for an image into a single mask. Filenames must match the image files exactly.

Example layout:

```text
restoration/
|-- restore.py
|-- data/
|   |-- single_patch/
|   |   |-- image_a.jpg
|   |   `-- detector_boxes.csv
|   `-- two_patches/
|       |-- image_b.jpg
|       `-- detector_boxes.csv
`-- runs/
```

## Install Dependencies

Install the core packages in the Python environment you use to run the script:

```bash
python -m pip install numpy opencv-python pillow
```

LaMa is optional:

```bash
python -m pip install simple-lama-inpainting
```

Stable Diffusion is optional. Install PyTorch appropriate for your machine, then install Diffusers and its dependencies:

```bash
python -m pip install torch diffusers transformers accelerate
```

If an optional method's dependencies are missing, the script prints an installation message and omits that method's latency from the summary. Stable Diffusion downloads its model the first time it runs, so it requires access to the model repository and may take time to initialize.

## Run

From the repository root:

```bash
python restoration/restore.py
```

Restored images are written to `restoration/runs`, in folders matching the input folders. Each source image produces a separate output for each available method.

## Latency Report

At the end, the script prints total execution time and a section for each option. Each section reports total measured latency and average latency per successful image for each input folder.

Option latency measures only its core operation: the pixel fill, OpenCV inpainting call, LaMa inference call, or Stable Diffusion pipeline call. It excludes input loading, mask creation, model setup/loading, output conversion and saving, and console output. The total execution time includes the full run, including setup and file I/O.