import os
import json
from ultralytics import YOLO

def select_yolo_model(model_dir, model_file=None):
    """Select or load YOLO model"""
    if not model_file:
        pt_files = [f for f in os.listdir(model_dir) if f.endswith('.pt')]
        if not pt_files:
            raise FileNotFoundError("No .pt files found in the model directory.")
        print("Available models:")
        for idx, f in enumerate(pt_files):
            print(f"{idx + 1}: {f}")
        choice = int(input("Select a model by number: ")) - 1
        model_file = pt_files[choice]

    return os.path.join(model_dir, model_file)

def load_class_info(model_file, category_mapping, cache_dir, use_cache):
    """Load or cache model class information, using category names from category_mapping"""
    cache_file = os.path.join(cache_dir, 'model', f"{os.path.splitext(os.path.basename(model_file))[0]}_classes.json")
    
    # If cache is enabled and the cache file exists, load and return cached class info
    if use_cache and os.path.exists(cache_file):
        print(f"Loading class info from cache: {cache_file}")
        with open(cache_file, "r") as f:
            return json.load(f)

    # Load the model and retrieve the list of class IDs
    model = YOLO(model_file)
    label_categories = []

    # Construct the label_categories list, containing each class's id and name
    for i, id_name in enumerate(model.names):
        category_id = i  # Retrieve category ID from the model
        category_name = category_mapping.get(i, id_name)  # Use the category_mapping to assign names; if absent, use the default model name
        label_categories.append({
            "id": category_id,
            "name": category_name
        })

    # Cache class information by creating the cache directory (if needed) and saving the JSON file
    os.makedirs(os.path.join(cache_dir, 'model'), exist_ok=True)
    with open(cache_file, "w") as f:
        json.dump(label_categories, f, indent=4)
    
    return label_categories