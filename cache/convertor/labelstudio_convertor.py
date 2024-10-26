import json
import uuid

def convert_to_json(image_path, image_file, sam_model_name, annotations):
    json_data = {
        "data": {
            "image": f"/data/local-files/?d=label-studio/data/localstorage/test/{image_file}"
        },
        "predictions": [
            {
                "model_version": sam_model_name,
                "result": []
            }
        ]
    }
    
    for annotation in annotations:
        bbox = annotation['bbox']
        mask = annotation['mask']
        img_width = annotation['img_width']
        img_height = annotation['img_height']
        category_id = annotation['category_id']
        category_name = annotation['category_name']
        
        x_min = bbox[0]
        y_min = bbox[1]
        width = bbox[2]
        height = bbox[3]
        
        x = (x_min / img_width) * 100
        y = (y_min / img_height) * 100
        w = (width / img_width) * 100
        h = (height / img_height) * 100
        
        annotation_id = str(uuid.uuid4())
        
        annotation_data = {
            "from_name": "label",
            "id": annotation_id,
            "source": "$image",
            "to_name": "image",
            "type": "rectanglelabels",
            "value": {
                "height": h,
                "rectanglelabels": [category_name],
                "rotation": 0,
                "width": w,
                "x": x,
                "y": y
            }
        }
        
        json_data["predictions"][0]["result"].append(annotation_data)
    
    return json_data