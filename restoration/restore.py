import csv
import cv2
import numpy as np
from pathlib import Path
from time import perf_counter


INPUT_WIDTH = 1280
INPUT_HEIGHT = 736
DETECTOR_BOX_FILE = "detector_boxes.csv"


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


def create_mask(image, detector_boxes):
    image_height, image_width = image.shape[:2]
    scale_x = image_width / INPUT_WIDTH
    scale_y = image_height / INPUT_HEIGHT
    mask = np.zeros((image_height, image_width), dtype=np.uint8)
    for x1, y1, x2, y2 in detector_boxes:
        mask[
            int(y1 * scale_y):int(y2 * scale_y),
            int(x1 * scale_x):int(x2 * scale_x),
        ] = 255
    return mask


def run_solid_fill(image, mask, output_path):
    start_time = perf_counter()
    result = image.copy()
    result[mask > 0] = (128, 128, 128)
    latency = perf_counter() - start_time
    cv2.imwrite(str(output_path), result)
    print("Option 1: Solid fill completed.")
    return latency


def run_telea_inpainting(image, mask, output_path):
    start_time = perf_counter()
    result = cv2.inpaint(image, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    latency = perf_counter() - start_time
    cv2.imwrite(str(output_path), result)
    print("Option 2: OpenCV Telea inpainting completed.")
    return latency


def run_lama_inpainting(image, mask, output_path):
    try:
        from PIL import Image
        from simple_lama_inpainting import SimpleLama
    except ImportError:
        print("Error: Install LaMa with 'pip install simple-lama-inpainting'")
        return None
    except OSError as error:
        print(f"Error: Could not load a required LaMa library: {error}")
        return None

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    pil_mask = Image.fromarray(mask)
    lama = SimpleLama()
    start_time = perf_counter()
    result_pil = lama(pil_image, pil_mask)
    latency = perf_counter() - start_time
    result = cv2.cvtColor(np.array(result_pil), cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(output_path), result)
    print("Option 3: LaMa deep learning inpainting completed.")
    return latency


def load_stable_diffusion_runtime():
    try:
        import torch
        from diffusers import StableDiffusionInpaintPipeline
        from PIL import Image
    except ImportError:
        print("Error: Install dependencies with 'pip install diffusers transformers accelerate torch'")
        return None
    except OSError as error:
        print(f"Error: Could not load a required Stable Diffusion library: {error}")
        return None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading Stable Diffusion on {device}... this might take a minute.")
    pipeline = StableDiffusionInpaintPipeline.from_pretrained(
        "sd2-community/stable-diffusion-2-inpainting",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        use_safetensors=True,
    ).to(device)
    return pipeline, Image


def run_stable_diffusion_inpainting(pipeline, image_class, image, mask, output_path):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = image_class.fromarray(image_rgb)
    pil_mask = image_class.fromarray(mask)

    width, height = pil_image.size
    width = (width // 8) * 8
    height = (height // 8) * 8
    start_time = perf_counter()
    result_pil = pipeline(
        prompt="high quality, seamless background",
        image=pil_image,
        mask_image=pil_mask,
        height=height,
        width=width,
    ).images[0]
    latency = perf_counter() - start_time

    result = cv2.cvtColor(np.array(result_pil), cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(output_path), result)
    print("Option 4: Stable Diffusion inpainting completed.")
    return latency


def load_migan_runtime(script_dir):
    try:
        import mi_gan
    except ImportError:
        print("Error: Install ONNX Runtime with 'pip install onnxruntime'")
        return None

    model_path = Path(mi_gan.MIGAN_MODEL_PATH)
    if not model_path.is_absolute():
        model_path = script_dir / model_path
    if not model_path.is_file():
        print(f"Downloading MI-GAN ONNX model to {model_path}...")
        mi_gan.download_migan_model(model_path)

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    ort_session = mi_gan.ort.InferenceSession(
        str(model_path),
        providers=providers,
    )
    print(f"Loaded MI-GAN model using providers: {ort_session.get_providers()}")
    return mi_gan, ort_session


def main():
    script_dir = Path(__file__).resolve().parent
    data_dir = script_dir / "data/attacked"
    output_dir = script_dir / "runs"
    output_dir.mkdir(parents=True, exist_ok=True)

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    folders = [path for path in sorted(data_dir.iterdir()) if path.is_dir()]
    folders.insert(0, data_dir)
    total_start = perf_counter()
    total_images = 0
    migan_runtime = None
    migan_setup_attempted = False
    stable_diffusion_runtime = None
    stable_diffusion_setup_attempted = False
    option_summaries = {
        "Option 01 - Solid fill": [],
        "Option 02 - Telea inpainting": [],
        "Option 03 - LaMa inpainting": [],
        "Option 04 - Stable Diffusion inpainting": [],
        "Option 05 - MI-GAN": [],
    }
    option_output_roots = {
        option: output_dir / option
        for option in option_summaries
    }

    for input_folder in folders:
        image_paths = sorted(
            path
            for path in input_folder.iterdir()
            if path.is_file() and path.suffix.lower() in image_extensions
        )
        if not image_paths:
            continue

        detector_boxes = load_detector_boxes(input_folder, image_paths)
        folder_output_dirs = {
            option: option_output_dir / input_folder.name
            if input_folder != data_dir
            else option_output_dir
            for option, option_output_dir in option_output_roots.items()
        }
        for folder_output_dir in folder_output_dirs.values():
            folder_output_dir.mkdir(parents=True, exist_ok=True)
        processed_images = 0
        folder_option_latencies = {option: 0.0 for option in option_summaries}
        folder_option_counts = {option: 0 for option in option_summaries}

        for image_path in image_paths:
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"Error: Could not load {image_path}")
                continue

            mask = create_mask(image, detector_boxes[image_path.name])
            input_stem = image_path.stem
            latency = run_solid_fill(
                image,
                mask,
                folder_output_dirs["Option 01 - Solid fill"]
                / f"restored_solid_fill_{input_stem}.jpg",
            )
            folder_option_latencies["Option 01 - Solid fill"] += latency
            folder_option_counts["Option 01 - Solid fill"] += 1

            latency = run_telea_inpainting(
                image,
                mask,
                folder_output_dirs["Option 02 - Telea inpainting"]
                / f"restored_inpainted_{input_stem}.jpg",
            )
            folder_option_latencies["Option 02 - Telea inpainting"] += latency
            folder_option_counts["Option 02 - Telea inpainting"] += 1

            latency = run_lama_inpainting(
                image,
                mask,
                folder_output_dirs["Option 03 - LaMa inpainting"]
                / f"restored_lama_{input_stem}.jpg",
            )
            if latency is not None:
                folder_option_latencies["Option 03 - LaMa inpainting"] += latency
                folder_option_counts["Option 03 - LaMa inpainting"] += 1

            if not stable_diffusion_setup_attempted:
                stable_diffusion_setup_attempted = True
                stable_diffusion_runtime = load_stable_diffusion_runtime()
            if stable_diffusion_runtime is not None:
                pipeline, image_class = stable_diffusion_runtime
                latency = run_stable_diffusion_inpainting(
                    pipeline,
                    image_class,
                    image,
                    mask,
                    folder_output_dirs["Option 04 - Stable Diffusion inpainting"]
                    / f"restored_sd_{input_stem}_diffusers.jpg",
                )
                folder_option_latencies["Option 04 - Stable Diffusion inpainting"] += latency
                folder_option_counts["Option 04 - Stable Diffusion inpainting"] += 1

            if not migan_setup_attempted:
                migan_setup_attempted = True
                migan_runtime = load_migan_runtime(script_dir)
            if migan_runtime is not None:
                migan, ort_session = migan_runtime
                migan_mask = migan.create_mask(
                    image,
                    detector_boxes[image_path.name],
                    apply_dilate=True,
                    kernel_size=7,
                )
                latency = migan.run_migan_inpainting(
                    ort_session,
                    image,
                    migan_mask,
                    folder_output_dirs["Option 05 - MI-GAN"]
                    / f"restored_migan_{input_stem}.jpg",
                )
                folder_option_latencies["Option 05 - MI-GAN"] += latency
                folder_option_counts["Option 05 - MI-GAN"] += 1
            processed_images += 1

        total_images += processed_images
        if processed_images:
            for option, latency in folder_option_latencies.items():
                count = folder_option_counts[option]
                if count:
                    option_summaries[option].append(
                        (input_folder.name, latency, latency / count)
                    )

    total_elapsed = perf_counter() - total_start
    print(
        f"All folders: {total_images} images, "
        f"total execution time {total_elapsed:.2f} s."
    )
    directory_width = max(
        [
            len("Directory"),
            *(
                len(directory_name)
                for summaries in option_summaries.values()
                for directory_name, _, _ in summaries
            ),
        ]
    )
    table_width = directory_width + 2 + 18 + 2 + 26
    for option, summaries in option_summaries.items():
        print(f"\n{option}")
        print(
            f"{'Directory':<{directory_width}}  "
            f"{'Total latency (s)':>18}  "
            f"{'Average latency (s/image)':>26}"
        )
        print("-" * table_width)
        for directory_name, directory_latency, average_latency in summaries:
            print(
                f"{directory_name:<{directory_width}}  "
                f"{directory_latency:>18.2f}  "
                f"{average_latency:>26.2f}"
            )

if __name__ == "__main__":
    main()