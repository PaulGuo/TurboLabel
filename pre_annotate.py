import os
import argparse
import importlib.util
import torch
import json
from core.config_loader import load_settings
from core.model_loader import select_yolo_model, load_class_info
from core.conversion_loader import generate_conversion_function
from core.annotation_processor import AnnotationProcessor
from core.numpy_encoder import NumpyEncoder
from ultralytics import YOLO

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    parser = argparse.ArgumentParser(description="Pre-annotation CLI Tool")
    parser.add_argument("--model", help="Specify YOLO model file (.pt)")
    parser.add_argument("--sam_checkpoint", default="./checkpoints/sam2.1_hiera_large.pt", help="Path to SAM model checkpoint")
    parser.add_argument("--sam_config", default="configs/sam2.1/sam2.1_hiera_l.yaml", help="Path to SAM model config")
    parser.add_argument("--output_format", default="labelstudio", help="Annotation output format (schema filename)")
    parser.add_argument("--use_cache", action="store_true", help="Use cached results if available")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()

    # Load configuration
    settings = load_settings()
    cache_folder = settings.get("paths")["cache_folder"]
    schema_folder = settings.get("paths")["schema_folder"]
    image_size = settings.get("processing")["image_size"]
    confidence_threshold = settings.get("processing")["confidence_threshold"]
    use_cache = settings.get("processing")["use_cache"]
    debug_mode = settings.get("processing")["debug_mode"]
    save_visualizations = settings.get("misc")["save_visualizations"]
    visualization_folder = settings.get("misc")["visualization_folder"]

    # Select or load YOLO model
    model_file = select_yolo_model("model", args.model)
    print(model_file)
    yolo_model = YOLO(model_file)
    category_mapping = settings.get("annotation")["category_mapping"]
    class_info = load_class_info(model_file, category_mapping, cache_folder, use_cache)
    print(class_info)

    # Generate and load conversion function
    conversion_path = generate_conversion_function(args.output_format, schema_folder, cache_folder, use_cache)
    spec = importlib.util.spec_from_file_location("conversion_function", conversion_path)
    conversion_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(conversion_module)
    conversion_function = conversion_module.convert_to_json

    # Load SAM2 model
    sam_model_name = settings.get("model")["sam_model_name"]
    sam_checkpoint = settings.get("model")["sam_model_checkpoint"]
    sam_config = settings.get("model")["sam_model_config"]
    annotation_processor = AnnotationProcessor(yolo_model, sam_model_name, sam_checkpoint, sam_config, device)

    # Process images
    input_folder = settings.get("paths")["input_folder"]
    output_folder = settings.get("paths")["output_folder"]
    os.makedirs(output_folder, exist_ok=True)

    for image_file in os.listdir(input_folder):
        if image_file.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_folder, image_file)
            annotations = annotation_processor.process_image(image_path, class_info, conversion_function, image_size, confidence_threshold, debug_mode, save_visualizations, visualization_folder)
            output_file = os.path.join(output_folder, f"{os.path.splitext(image_file)[0]}.json")
            with open(output_file, "w") as f:
                print(annotations)
                json.dump(annotations, f, indent=4, cls=NumpyEncoder)
            print(f"Saved annotations to {output_file}")

if __name__ == "__main__":
    main()