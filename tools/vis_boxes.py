import cv2
import yaml
import matplotlib.pyplot as plt
import os

def visualize_bboxes(image_id, bbox_info_path='bbox_info.yaml'):
    with open(bbox_info_path, 'r') as f:
        bbox_info = yaml.load(f, Loader=yaml.Loader)
    
    # Filter bounding boxes for the given image_id
    filtered_bboxes = [info for info in bbox_info if info["image_id"] == image_id]
    
    if not filtered_bboxes:
        print(f"No bounding boxes found for image_id: {image_id}")
        return
    
    image_path = filtered_bboxes[0]["image_path"]
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    for info in filtered_bboxes:
        bbox = info["bbox"]
        entity_id = info["entity_id"]
        
        # Draw bounding box
        cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color=(255, 0, 0), thickness=2)
        
        # Put entity id text
        cv2.putText(image, str(entity_id), (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Display the image
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    image_id = 0  # change to the desired image id
    visualize_bboxes(image_id)
