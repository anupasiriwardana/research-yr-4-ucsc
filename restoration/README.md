# Image Restoration

This folder contains three scripts for restoring image regions marked by
detector boxes:

- `restore.py` runs solid fill, OpenCV Telea, LaMa, and Stable Diffusion.
- `precise_telea.py` runs a single OpenCV inpainting method. Despite its
  filename and output labels, its current implementation uses OpenCV's
  Navier-Stokes method (`cv2.INPAINT_NS`) with an inpainting radius of 7.
- `mi_gan.py` runs MI-GAN using an ONNX pipeline.

## Input Images and Detector Boxes

Put images under `data`, either directly in that folder or in its immediate
subfolders. Supported formats are `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tif`, and
`.tiff`. Each folder containing images needs its own `detector_boxes.csv`.

The CSV maps filenames to one or more boxes in the 1280x736 detector-input
coordinate system. Use one row per box, repeating the filename when an image
has multiple boxes:

```csv
filename,x1,y1,x2,y2
image_a.jpg,100,120,180,200
image_a.jpg,400,220,480,300
image_b.jpg,520,250,600,330
```

Coordinates must satisfy `0 <= x1 < x2 <= 1280` and
`0 <= y1 < y2 <= 736`. Every image needs at least one row, and each filename
must match its image file exactly. The scripts scale the boxes to each image's
dimensions and combine them into a mask. `precise_telea.py` and `mi_gan.py`
dilate their masks before inpainting.

Example layout:

```text
restoration/
|-- restore.py
|-- precise_telea.py
|-- mi_gan.py
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

Install the core dependencies for `restore.py` and `precise_telea.py`:

```bash
python -m pip install numpy opencv-python
```

`restore.py`'s LaMa option is optional:

```bash
python -m pip install simple-lama-inpainting
```

Stable Diffusion is also optional. Install a PyTorch build appropriate for your
machine, then install Diffusers and its dependencies:

```bash
python -m pip install torch diffusers transformers accelerate
```

For MI-GAN, install ONNX Runtime:

```bash
python -m pip install onnxruntime
```

To use ONNX Runtime with CUDA, install the compatible `onnxruntime-gpu` package
instead of `onnxruntime` and configure its CUDA dependencies. The script tries
CUDA first and falls back to CPU when CUDA is unavailable.

## Run

Run commands from the repository root:

```bash
python restoration/restore.py
python restoration/precise_telea.py
python restoration/mi_gan.py
```

All scripts use `restoration/data` for input and write results below
`restoration/runs`, keeping separate subfolders for input folders:

- `restore.py`: `restoration/runs/`
- `precise_telea.py`: `restoration/runs/precise_telea/`
- `mi_gan.py`: `restoration/runs/migan/`

The scripts report total run time and per-folder inpainting latency. The
per-folder averages are measured per successfully processed image.

## MI-GAN Model

On its first run, `mi_gan.py` downloads the official
[MI-GAN ONNX pipeline](https://huggingface.co/andraniksargsyan/migan/blob/main/migan_pipeline_v2.onnx)
to `restoration/migan_pipeline_v2.onnx` and verifies its SHA-256 checksum.
Internet access is needed for this initial download; later runs reuse the local
file. The model file is ignored by Git.

The script constructs an inpainting mask from the detector boxes, dilates it,
and converts it to the MI-GAN pipeline's expected convention before inference.

## Latency Measurements

The latency scope differs slightly by script:

- `restore.py` times each method's main fill or inference operation.
- `precise_telea.py` times the OpenCV inpainting call.
- `mi_gan.py` times preprocessing, ONNX inference, and postprocessing.

These measurements exclude input loading, mask creation, and output saving.
Model setup and download are also excluded from per-image latency but included
in total run time.
