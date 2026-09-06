import os
import cv2
from gtts import gTTS
from IPython.display import Audio, display
from ultralytics import YOLO
from PIL import Image
import matplotlib.pyplot as plt

# Load YOLOv10 Model
model = YOLO("weights/yolov10s.pt")


def resize_image(image_path, target_size=(640, 640)):
    """Resizes an image to the given target size."""
    image = cv2.imread(image_path)
    resized_image = cv2.resize(image, target_size)  # Resize the image
    resized_path = "/resized_image.jpg"
    cv2.imwrite(resized_path, resized_image)  # Save resized image
    return resized_path, resized_image.shape  # Return new path & dimensions

def detect_objects(image_path):
    # Load YOLO model
    resized_path, image_shape = resize_image(image_path)
    image_width, image_height = image_shape[1], image_shape[0]
    ## Load YOLOv10 Model try other waits in later stage
    #model = YOLO("weights/yolov10s.pt")


    # Perform object detection
    results = model.predict(resized_path, save=True)

    # Initialize descriptions
    environment_description = []
    detected_anything = False
    object_positions = {"left": [], "center": [], "right": []}

    # Image size (assuming all detections have the same image width)
    image_width = results[0].orig_shape[1]  # Get the width of the image

    # Define region thresholds
    left_boundary = image_width * 0.33
    right_boundary = image_width * 0.66

    # Process each result
    for result in results:
        detected_boxes = result.boxes
        detected_classes = detected_boxes.cls
        detected_confidences = detected_boxes.conf
        detected_xyxy = detected_boxes.xyxy  # (x_min, y_min, x_max, y_max)

        object_names = [result.names[int(cls.item())] for cls in detected_classes]
        threshold = 0.5
        filtered_objects = [(obj, conf.item(), xyxy) for obj, conf, xyxy in zip(object_names, detected_confidences, detected_xyxy) if conf.item() >= threshold]

        if filtered_objects:
            detected_anything = True
            for obj, _, xyxy in filtered_objects:
                x_min, _, x_max, _ = xyxy.tolist()

                # Classify object positions
                if x_max < left_boundary:
                    object_positions["left"].append(obj)
                elif x_min > right_boundary:
                    object_positions["right"].append(obj)
                else:
                    object_positions["center"].append(obj)

    # Construct environment description
    if detected_anything:
        for position, objs in object_positions.items():
            if objs:
                obj_counts = {obj: objs.count(obj) for obj in set(objs)}
                obj_list = [f"{count} {obj}{'s' if count > 1 else ''}" for obj, count in obj_counts.items()]
                environment_description.append(f"{', '.join(obj_list)} on the {position}.")
    else:
        environment_description.append("No significant objects detected.")

    # Debugging: Print detected objects and their positions
    print("\n**Detected Objects by Position:**")
    print(f"Left: {object_positions['left']}")
    print(f"Center: {object_positions['center']}")
    print(f"Right: {object_positions['right']}")

    # Stop execution if no object is detected
    if not detected_anything:
        print("No objects detected. Stopping execution.")
        return

    # Create full description
    full_description = " ".join(environment_description)

    # Convert text to speech
    tts = gTTS(text=full_description, lang='en')
    tts.save("environment_description.mp3")

    # Play the audio
    display(Audio("environment_description.mp3", autoplay=True))

# Run the function
image_path = "testing_images\im1.jpg"
detect_objects(image_path)

