import json
import os
from PIL import Image

# Set the path to the folder where txt files and images are located
txt_folder = ''
img_folder = ''
# Output JSON file path
output_json_path = 'output.json'

# Initialize the COCO data structure
coco_data = {
    "images": [],
    "annotations": [],
    "categories": []
}

# Category mapping, adapted to your reality
category_mapping = {0: '',1:''}
for category_id, category_name in category_mapping.items():
    coco_data["categories"].append({
        "id": category_id + 1,  # COCO category IDs usually start at 1
        "name": category_name
    })

annotation_id = 1  # Initialize the annotation ID

for filename in os.listdir(txt_folder):
    if filename.endswith('.txt'):
        image_id = int(filename.split('.')[0])
        img_path = os.path.join(img_folder, filename.replace('.txt', '.jpg'))
        with Image.open(img_path) as img:
            img_width, img_height = img.size

        # Adding Picture Information to COCO Data
        coco_data["images"].append({
            "id": image_id,
            "file_name": os.path.basename(img_path),
            "width": img_width,
            "height": img_height
        })

        txt_path = os.path.join(txt_folder, filename)
        with open(txt_path, 'r') as file:
            for line in file:
                parts = line.strip().split()
                category_id, x_center, y_center, width, height = map(float, parts)

                # Convert relative coordinates and dimensions to absolute values
                abs_width = width * img_width
                abs_height = height * img_height
                x_min = (x_center * img_width) - (abs_width / 2)
                y_min = (y_center * img_height) - (abs_height / 2)

                coco_data["annotations"].append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": int(category_id) + 1,
                    "bbox": [x_min, y_min, abs_width, abs_height],
                    "area": abs_width * abs_height,
                    "iscrowd": 0
                })
                annotation_id += 1

# Write to JSON file
with open(output_json_path, 'w') as json_file:
    json.dump(coco_data, json_file)

print("Conversion completed!")
