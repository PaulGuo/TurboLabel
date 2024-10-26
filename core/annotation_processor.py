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

    def calculate_iou(self, box1, box2):
        """Calculate Intersection over Union (IoU) between two bounding boxes in xywh format"""
        # Convert [x, y, w, h] to [x_min, y_min, x_max, y_max]
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2

        x1_max, y1_max = x1 + w1, y1 + h1
        x2_max, y2_max = x2 + w2, y2 + h2

        # Compute the intersection coordinates
        xi1, yi1 = max(x1, x2), max(y1, y2)
        xi2, yi2 = min(x1_max, x2_max), min(y1_max, y2_max)

        # Compute area of intersection
        inter_width = max(0, xi2 - xi1)
        inter_height = max(0, yi2 - yi1)
        inter_area = inter_width * inter_height

        # Compute areas of each box and union area
        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - inter_area

        # Calculate and return IoU
        return inter_area / union_area if union_area > 0 else 0

    def check_duplicate_bboxes(self, annotations, image_file):
        """Check for duplicate bounding boxes with high IoU and log them."""
        duplicate_log = []
        for i in range(len(annotations)):
            for j in range(i + 1, len(annotations)):
                bbox1 = annotations[i]['bbox']
                bbox2 = annotations[j]['bbox']
                iou = self.calculate_iou(bbox1, bbox2)

                # If IoU is high (greater than 0.99), log as duplicate
                if iou > 0.94:
                    category_name_bbox1 = annotations[i]['category_name']
                    category_name_bbox2 = annotations[j]['category_name']
                    duplicate_log.append(f"Image: {image_file}, Category: {category_name_bbox1}, BBox1: {bbox1}, Category: {category_name_bbox2}, BBox2: {bbox2}")

        # Write duplicates to log file if any were found
        if duplicate_log:
            log_file_path = "logs/duplicate_bboxes.log"
            os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
            with open(log_file_path, "a") as log_file:
                for log in duplicate_log:
                    log_file.write(log + "\n")

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

        # Check for duplicate bounding boxes based on IoU
        self.check_duplicate_bboxes(annotations, image_file)

        # If in debug mode, display visualization output
        if debug_mode and debug_boxes:
            visualize_annotations(image_name, image, debug_boxes, debug_labels, debug_masks, np.vstack(debug_points), np.concatenate(debug_point_labels), save_visualizations, visualization_folder)

        # Generate output format using the conversion function
        return conversion_function(image_path, image_file, self.sam_model_name, annotations)