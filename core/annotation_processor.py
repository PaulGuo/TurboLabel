import numpy as np
import json
import uuid
import os
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from core.visualize_annotations import visualize_annotations

class AnnotationProcessor:
    def __init__(self, yolo_model, sam_model_name, sam2_checkpoint, sam2_config, device):
        self.model = yolo_model
        self.sam_model_name = sam_model_name
        self.predictor = SAM2ImagePredictor(build_sam2(sam2_config, sam2_checkpoint, device=device))

    def process_image(self, image_path, class_info, conversion_function, image_size, confidence_threshold, debug_mode, save_visualizations, visualization_folder):
        """Process a single image and generate annotations"""
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        image_extension = os.path.splitext(image_path)[1]
        image_file = f"{image_name}{image_extension}"
        image = Image.open(image_path).convert("RGB")
        results = self.model.predict(image_path, imgsz=image_size, conf=confidence_threshold)
        boxes = results[0].boxes

        image = Image.open(image_path)
        image = np.array(image.convert("RGB"))
        self.predictor.set_image(image)
        annotations = []

        if debug_mode:
            debug_boxes = []
            debug_labels = []
            debug_masks = []
            debug_points = []
            debug_point_labels = []

        for box in boxes:
            category_id = int(box.cls)  # Get category ID
            confidence = box.conf  # Get confidence score

            if confidence >= confidence_threshold:
                # Use xyxy to get the bounding box coordinates
                x_min, y_min, x_max, y_max = box.xyxy[0].tolist()  # Retrieve bounding box boundaries using box.xyxy

                # Calculate the center point of the bounding box
                center_x = (x_min + x_max) / 2
                center_y = (y_min + y_max) / 2
                input_point = np.array([[center_x, center_y]])
                input_label = np.array([1])

                # Perform segmentation using SAM2
                masks, scores, _ = self.predictor.predict(
                    point_coords=input_point,
                    point_labels=input_label,
                    multimask_output=True
                )

                # Select the mask with the highest confidence score
                max_score_idx = np.argmax(scores)
                best_mask = masks[max_score_idx]

                # Calculate the bounding box of the best mask
                ys, xs = np.where(best_mask)
                x_min, x_max = xs.min(), xs.max()
                y_min, y_max = ys.min(), ys.max()

                # Convert to relative coordinates and generate annotation data
                img_height, img_width = image.shape[:2]
                category_name = class_info[category_id]['name']

                annotation = {
                    "id": str(uuid.uuid4()),
                    "bbox": [x_min, y_min, x_max - x_min, y_max - y_min], # x,y,w,h
                    "mask": best_mask.tolist(),
                    "img_width": img_width,
                    "img_height": img_height,
                    "category_id": category_id,
                    "category_name": category_name,
                }
                annotations.append(annotation)

                if debug_mode:
                    debug_boxes.append((x_min, y_min, x_max, y_max))
                    debug_labels.append(category_id)
                    debug_masks.append(best_mask)
                    debug_points.append(input_point)
                    debug_point_labels.append(input_label)

        # If in debug mode, display visualization output
        if debug_mode and debug_boxes:
            visualize_annotations(image_name, image, debug_boxes, debug_labels, debug_masks, np.vstack(debug_points), np.concatenate(debug_point_labels), save_visualizations, visualization_folder)

        # Generate output format using the conversion function
        return conversion_function(image_path, image_file, self.sam_model_name, annotations)