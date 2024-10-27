import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

def show_mask(mask, ax, random_color=False, borders=True):
    """Display the SAM2 mask with optional random color and borders."""
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)  # Random RGBA color
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])  # Default blue color with transparency

    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)

    if borders:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # Smooth contours for better appearance
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2)
    
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=175):
    """Display center points with green for positive labels and red for negative labels."""
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

def visualize_annotations(image_name, image, boxes, labels, masks, points, point_labels, save_visualizations, visualization_folder, original_boxes=None):
    """Visualize YOLO bounding boxes, class labels, SAM2 masks, and center points."""
    height, width = image.shape[:2]
    plt.figure(figsize=(width / 100, height / 100), dpi=100)  # Scale figsize and dpi to match image size
    plt.imshow(image)

    # Remove any white border or padding in the figure
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # Display SAM2 masks
    for mask in masks:
        show_mask(mask, plt.gca())

    # Display YOLO bounding boxes and class labels (SAM adjusted)
    for box, label in zip(boxes, labels):
        x_min, y_min, x_max, y_max = box
        plt.gca().add_patch(plt.Rectangle(
            (x_min, y_min), x_max - x_min, y_max - y_min,
            edgecolor='red', facecolor='none', linewidth=2))
        plt.text(x_min, y_min - 5, f'{label}', color='red', fontsize=12)

    # Optionally display original YOLO detection boxes as dashed rectangles
    if original_boxes is not None:
        for box in original_boxes:
            # Extract coordinates in [x_min, y_min, x_max, y_max] format
            x_min, y_min, x_max, y_max = box.xyxy[0].tolist()
            plt.gca().add_patch(plt.Rectangle(
                (x_min, y_min), x_max - x_min, y_max - y_min,
                edgecolor='yellow', facecolor='none', linewidth=1, linestyle='--'))

    # Display center points
    show_points(points, point_labels, plt.gca())

    plt.axis('off')

    # Save or display the visualization
    if save_visualizations:
        os.makedirs(visualization_folder, exist_ok=True)
        file_path = os.path.join(visualization_folder, f"{image_name}.png")
        plt.savefig(file_path)
        print(f"Visualization saved to {file_path}")
        plt.close()
    else:
        plt.show()