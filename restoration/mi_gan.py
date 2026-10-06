import csv
import cv2
import hashlib
import numpy as np
import onnxruntime as ort
from pathlib import Path
import tempfile
from time import perf_counter
from urllib.request import urlopen

INPUT_WIDTH = 1280
INPUT_HEIGHT = 736
DETECTOR_BOX_FILE = "detector_boxes.csv"
MIGAN_MODEL_PATH = "migan_pipeline_v2.onnx"
MIGAN_MODEL_URL = (
    "https://huggingface.co/andraniksargsyan/migan/resolve/main/"
    "migan_pipeline_v2.onnx"
)
MIGAN_MODEL_SHA256 = (
    "6f1f3530a1a2324b19752018ce756088b07973cda8d7d890034ace5c8a48c40b"
)

def load_detector_boxes(folder, image_paths):
    boxes_path = folder / DETECTOR_BOX_FILE
    if not boxes_path.is_file():
        raise FileNotFoundError(
            f"Missing {boxes_path}. Add one row per box with columns "
            "filename,x1,y1,x2,y2."
        )

    boxes = {}
    with boxes_path.open(newline="", encoding="utf-8-sig") as boxes_file:
        reader = csv.DictReader(boxes_file)
        required_columns = {"filename", "x1", "y1", "x2", "y2"}
        if not reader.fieldnames or not required_columns.issubset(reader.fieldnames):
            raise ValueError(
                f"{boxes_path} must have columns: filename,x1,y1,x2,y2."
            )

        for row in reader:
            filename = (row.get("filename") or "").strip()
            try:
                box = tuple(int(row[key]) for key in ("x1", "y1", "x2", "y2"))
            except (TypeError, ValueError):
                raise ValueError(
                    f"Invalid or blank box coordinates for '{filename}' "
                    f"in {boxes_path} (CSV row {reader.line_num})."
                ) from None

            x1, y1, x2, y2 = box
            if not filename or not (0 <= x1 < x2 <= INPUT_WIDTH and 0 <= y1 < y2 <= INPUT_HEIGHT):
                raise ValueError(
                    f"Invalid filename or box coordinates for CSV row "
                    f"{reader.line_num} in {boxes_path}. Coordinates must fit "
                    f"within {INPUT_WIDTH}x{INPUT_HEIGHT}."
                )
            boxes.setdefault(filename, []).append(box)

    missing_filenames = [path.name for path in image_paths if not boxes.get(path.name)]
    if missing_filenames:
        raise ValueError(
            f"Missing box entries in {boxes_path} for: "
            f"{', '.join(missing_filenames)}."
        )
    return boxes

def create_mask(image, detector_boxes, apply_dilate=True, kernel_size=7, iterations=1):
    image_height, image_width = image.shape[:2]
    
    scale_x = image_width / INPUT_WIDTH
    scale_y = image_height / INPUT_HEIGHT
    mask = np.zeros((image_height, image_width), dtype=np.uint8)
    
    for x1, y1, x2, y2 in detector_boxes:
        mask[
            int(y1 * scale_y):int(y2 * scale_y),
            int(x1 * scale_x):int(x2 * scale_x),
        ] = 255
        
    if apply_dilate:
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=iterations)
        
    return mask

def preprocess_for_migan(image, mask):
    orig_w, orig_h = image.shape[1], image.shape[0]
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_tensor = np.expand_dims(
        np.ascontiguousarray(np.transpose(img_rgb, (2, 0, 1))), axis=0
    )
    known_mask = cv2.bitwise_not(mask)
    mask_tensor = np.expand_dims(known_mask, axis=(0, 1))
    return img_tensor, mask_tensor, (orig_w, orig_h)

def postprocess_migan_output(output_tensor, orig_size):
    out_img = np.squeeze(output_tensor, axis=0)
    out_img = np.transpose(out_img, (1, 2, 0))
    out_bgr = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)
    final_img = cv2.resize(out_bgr, orig_size)
    return final_img

def download_migan_model(model_path):
    model_path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=model_path.parent, suffix=".download", delete=False
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            with urlopen(MIGAN_MODEL_URL, timeout=60) as response:
                while chunk := response.read(1024 * 1024):
                    digest.update(chunk)
                    temporary_file.write(chunk)

        if digest.hexdigest() != MIGAN_MODEL_SHA256:
            raise ValueError(
                "Downloaded MI-GAN model failed SHA-256 verification."
            )
        temporary_path.replace(model_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()

def run_migan_inpainting(ort_session, image, mask, output_path):
    start_time = perf_counter()
    
    img_tensor, mask_tensor, orig_size = preprocess_for_migan(image, mask)
    
    # Run inference
    input_name_img = ort_session.get_inputs()[0].name
    input_name_mask = ort_session.get_inputs()[1].name
    
    outputs = ort_session.run(None, {
        input_name_img: img_tensor, 
        input_name_mask: mask_tensor
    })
    
    # Postprocess
    result = postprocess_migan_output(outputs[0], orig_size)
    
    latency = perf_counter() - start_time
    cv2.imwrite(str(output_path), result)
    print("MI-GAN inpainting completed.")
    return latency

def main():
    script_dir = Path(__file__).resolve().parent
    data_dir = script_dir / "data"
    output_dir = script_dir / "runs/migan"
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = Path(MIGAN_MODEL_PATH)
    if not model_path.is_absolute():
        model_path = script_dir / model_path
    if not model_path.is_file():
        print(f"Downloading MI-GAN ONNX model to {model_path}...")
        download_migan_model(model_path)
    
    # Use CUDAExecutionProvider if you have an NVIDIA GPU, otherwise CPUExecutionProvider
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    ort_session = ort.InferenceSession(str(model_path), providers=providers)
    print(f"Loaded MI-GAN model using providers: {ort_session.get_providers()}")

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    folders = [path for path in sorted(data_dir.iterdir()) if path.is_dir()]
    folders.insert(0, data_dir)
    total_start = perf_counter()
    total_images = 0
    folder_summaries = []

    for input_folder in folders:
        image_paths = sorted(
            path
            for path in input_folder.iterdir()
            if path.is_file() and path.suffix.lower() in image_extensions
        )
        if not image_paths:
            continue

        detector_boxes = load_detector_boxes(input_folder, image_paths)
        folder_output_dir = output_dir / input_folder.name if input_folder != data_dir else output_dir
        folder_output_dir.mkdir(parents=True, exist_ok=True)
        processed_images = 0
        folder_latency = 0.0

        for image_path in image_paths:
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"Error: Could not load {image_path}")
                continue

            # Note: I kept kernel_size=7 as recommended earlier for larger patches
            mask = create_mask(image, detector_boxes[image_path.name], apply_dilate=True, kernel_size=7)
            input_stem = image_path.stem
            
            latency = run_migan_inpainting(
                ort_session,
                image,
                mask,
                folder_output_dir / f"restored_migan_{input_stem}.jpg",
            )
            folder_latency += latency
            processed_images += 1

        total_images += processed_images
        if processed_images:
            folder_summaries.append(
                (input_folder.name, folder_latency, folder_latency / processed_images)
            )

    total_elapsed = perf_counter() - total_start
    print(
        f"All folders: {total_images} images, "
        f"total execution time {total_elapsed:.2f} s."
    )
    directory_width = max(
        [len("Directory"), *(len(name) for name, _, _ in folder_summaries)]
    )
    table_width = directory_width + 2 + 18 + 2 + 26
    print("\nMI-GAN inpainting")
    print(
        f"{'Directory':<{directory_width}}  "
        f"{'Total latency (s)':>18}  "
        f"{'Average latency (s/image)':>26}"
    )
    print("-" * table_width)
    for directory_name, directory_latency, average_latency in folder_summaries:
        print(
            f"{directory_name:<{directory_width}}  "
            f"{directory_latency:>18.2f}  "
            f"{average_latency:>26.2f}"
        )

if __name__ == "__main__":
    main()