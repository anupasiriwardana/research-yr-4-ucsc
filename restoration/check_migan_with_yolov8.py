from ultralytics import YOLO
from pathlib import Path

script_dir = Path(__file__).resolve().parent
source_dir = script_dir / 'runs' / 'migan'
output_dir = script_dir / 'runs' / 'yolov8_detections'
image_extensions = {'.bmp', '.dng', '.jpeg', '.jpg', '.mpo', '.png', '.tif', '.tiff', '.webp', '.pfm'}
images_by_directory = {}

for image_path in sorted(source_dir.rglob('*')):
    if image_path.is_file() and image_path.suffix.lower() in image_extensions:
        relative_directory = image_path.parent.relative_to(source_dir)
        images_by_directory.setdefault(relative_directory, []).append(image_path)

if not images_by_directory:
    raise FileNotFoundError(f'No supported images found under {source_dir}')

# 1. Load the model (replace 'yolov8n.pt' with your custom .pt weights if needed)
model = YOLO(str(script_dir / 'yolov8n.pt'))

for relative_directory, image_paths in images_by_directory.items():
    output_name = Path(output_dir.name) / relative_directory
    results = model.predict(
        source=[str(image_path) for image_path in image_paths],
        save=True,
        save_txt=True,
        save_crop=True,
        project=str(output_dir.parent),
        name=str(output_name),
        exist_ok=True,
    )
    print(f'Processed {len(results)} image(s): {output_dir / relative_directory}')