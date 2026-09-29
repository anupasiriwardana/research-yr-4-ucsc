import cv2
import numpy as np
from pathlib import Path


INPUT_WIDTH = 1280
INPUT_HEIGHT = 736
DETECTOR_BOX = (500, 252, 568, 324)


def create_mask(image):
    image_height, image_width = image.shape[:2]
    scale_x = image_width / INPUT_WIDTH
    scale_y = image_height / INPUT_HEIGHT
    x1, y1, x2, y2 = DETECTOR_BOX

    mask = np.zeros((image_height, image_width), dtype=np.uint8)
    mask[
        int(y1 * scale_y):int(y2 * scale_y),
        int(x1 * scale_x):int(x2 * scale_x),
    ] = 255
    return mask


def run_solid_fill(image, mask, output_path):
    result = image.copy()
    result[mask > 0] = (128, 128, 128)
    cv2.imwrite(str(output_path), result)
    print("Option 1: Solid fill completed.")


def run_telea_inpainting(image, mask, output_path):
    result = cv2.inpaint(image, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    cv2.imwrite(str(output_path), result)
    print("Option 2: OpenCV Telea inpainting completed.")


def run_lama_inpainting(image, mask, output_path):
    try:
        from PIL import Image
        from simple_lama_inpainting import SimpleLama
    except ImportError:
        print("Error: Install LaMa with 'pip install simple-lama-inpainting'")
        return
    except OSError as error:
        print(f"Error: Could not load a required LaMa library: {error}")
        return

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    pil_mask = Image.fromarray(mask)
    result_pil = SimpleLama()(pil_image, pil_mask)
    result = cv2.cvtColor(np.array(result_pil), cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(output_path), result)
    print("Option 3: LaMa deep learning inpainting completed.")


def run_stable_diffusion_inpainting(image, mask, output_path):
    try:
        import torch
        from diffusers import StableDiffusionInpaintPipeline
        from PIL import Image
    except ImportError:
        print("Error: Install dependencies with 'pip install diffusers transformers accelerate torch'")
        return
    except OSError as error:
        print(f"Error: Could not load a required Stable Diffusion library: {error}")
        return

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    pil_mask = Image.fromarray(mask)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading Stable Diffusion on {device}... this might take a minute.")
    pipeline = StableDiffusionInpaintPipeline.from_pretrained(
        "sd2-community/stable-diffusion-2-inpainting",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        use_safetensors=True,
    ).to(device)

    width, height = pil_image.size
    width = (width // 8) * 8
    height = (height // 8) * 8
    result_pil = pipeline(
        prompt="high quality, seamless background",
        image=pil_image,
        mask_image=pil_mask,
        height=height,
        width=width,
    ).images[0]

    result = cv2.cvtColor(np.array(result_pil), cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(output_path), result)
    print("Option 4: Stable Diffusion inpainting completed.")


def main():
    script_dir = Path(__file__).resolve().parent
    image_path = script_dir / "data" / "ccc43019-54a0931a.jpg"
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Error: Could not load {image_path}")
        return

    output_dir = script_dir / "runs"
    output_dir.mkdir(parents=True, exist_ok=True)
    mask = create_mask(image)
    input_stem = image_path.stem

    run_solid_fill(image, mask, output_dir / "restored_solid_fill_1.jpg")
    run_telea_inpainting(
        image,
        mask,
        output_dir / f"restored_inpainted_{input_stem}.jpg",
    )

    run_lama_inpainting(image, mask, output_dir / f"restored_lama_{input_stem}.jpg")
    run_stable_diffusion_inpainting(
        image,
        mask,
        output_dir / f"restored_sd_{input_stem}_diffusers.jpg",
    )

if __name__ == "__main__":
    main()