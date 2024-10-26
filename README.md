# TurboLabel

#### A powerful annotation tool combining YOLO, SAM, and LLMs for flexible image labeling.

**TurboLabel** is an advanced annotation tool that combines powerful object detection with SAM-based segmentation to automate and streamline image labeling. Designed for flexible compatibility, TurboLabel supports multiple annotation formats such as Label Studio and COCO, and offers efficient caching, model selection, and flexible configuration for various use cases.

## Features

- **Automated Annotation**: Combines YOLO-based object detection with SAM segmentation to generate accurate bounding boxes and masks.
- **Flexible Formats with LLM Support**: Leverages Large Language Models (LLMs) to dynamically support various annotation formats. This enables TurboLabel to automatically adapt to schema requirements, including Label Studio, COCO, and custom formats.
- **Caching**: Includes caching options to optimize performance, especially for repeated tasks.
- **Configurable Settings**: Use `settings.yaml` to adjust confidence thresholds, paths, model settings, and annotation formats.
- **CLI-Based**: A command-line interface (CLI) to streamline the labeling process with options for model selection, debugging, and cache management.

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/PaulGuo/TurboLabel.git
   cd TurboLabel
   ```

2. **Install required packages:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Download required checkpoints:**

   ```bash
   ./sam2/checkpoints/download_ckpts.sh
   ```

4. **Configure settings:**

   Adjust the configurations in `settings.yaml` to match your environment and project needs. 

5. **Prepare model and data files:**

   Place the pretrained YOLO model file in the `model` directory and the images you want to annotate in the `data` directory.

## Usage

### Example Command

```bash
python pre_annotate.py
```

## Configuration

Edit `settings.yaml` to customize the following options:

- **model**: Specify paths for YOLO and SAM models, as well as configuration files.
- **paths**: Define input/output directories and cache locations.
- **processing**: Set confidence thresholds, image sizes, and cache options.
- **annotation**: Choose the annotation format and category mappings.
- **misc**: Enable or disable visualization and debug mode options.

TurboLabel’s configuration file (`settings.yaml`) supports the following parameters:

- **model**: 
  - **yolo_model_path** (str): Path to the YOLO model file used for annotation.
  - **sam_model_name** (str): Name of the SAM model to be used.
  - **sam_model_checkpoint** (str): Path to the SAM model checkpoint file.
  - **sam_model_config** (str): Path to the SAM model configuration file.

- **paths**: 
  - **input_folder** (str): Directory path for the input images.
  - **output_folder** (str): Directory path for saving the generated annotations.
  - **cache_folder** (str): Directory path for cached files.
  - **schema_folder** (str): Directory path for schema files used in different annotation formats.

- **processing**: 
  - **confidence_threshold** (float): Confidence threshold for YOLO predictions.
  - **image_size** (int): Image size to be used for model processing.
  - **use_cache** (bool): Whether to use cached data for faster processing.
  - **debug_mode** (bool): Enables debug mode, providing additional output for troubleshooting.

- **annotation**:
  - **format** (str): Desired annotation format (e.g., `labelstudio`, `coco`) for output files.
  - **category_mapping** (dict): Mapping of category IDs to category names for annotations.

- **misc**: 
  - **save_visualizations** (bool): Whether to save visualizations of annotations.
  - **visualization_folder** (str): Directory path for storing visualization files if enabled.

These parameters provide flexibility and control over TurboLabel’s behavior, allowing you to tailor the tool for different models, annotation formats, and debugging preferences.

## Project Structure

```
├── LICENSE               # License file for the project
├── README.md             # Project description, setup, and usage instructions
├── annotations           # Directory for storing generated annotation files
├── cache                 # Directory for caching precomputed results and model data
├── core                  # Core module with main functionality and helper scripts
├── data                  # Directory for input images that need annotation
├── docs                  # Documentation files for the project
├── model                 # Directory for storing model files (e.g., pretrained YOLO models)
├── sam2                  # SAM model configurations, checkpoints, and related files
├── schema                # Directory for annotation schema files supporting various formats
├── ultralytics           # YOLO model configurations and additional dependencies
├── visualizations        # Directory for saving visualizations of annotations (if enabled)
├── pre_annotate.py       # Main script to run the annotation process
├── requirements.txt      # List of Python dependencies for the project
├── settings.yaml         # Main configuration file for general settings
└── pyproject.toml        # Project metadata and build configuration
```

## Contribution

Contributions are welcome! Feel free to open issues or submit pull requests to improve TurboLabel.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.