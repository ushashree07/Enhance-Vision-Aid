import cv2
import torch
from ultralytics import YOLO
from gtts import gTTS
from IPython.display import Audio, display

# Load YOLO model
model = YOLO("weights\yolov10m.pt")    # Ensure the correct path to weights

def classify_object_positions(results, frame_width):
    """Classifies detected objects into left, center, or right."""
    object_positions = {"left": [], "center": [], "right": []}
    detected_anything = False

    # Define region thresholds
    left_boundary = frame_width * 0.40
    right_boundary = frame_width * 0.60

    for result in results:
        detected_boxes = result.boxes
        detected_classes = detected_boxes.cls
        detected_confidences = detected_boxes.conf
        detected_xyxy = detected_boxes.xyxy

        object_names = [result.names[int(cls.item())] for cls in detected_classes]
        threshold = 0.3
        filtered_objects = [(obj, conf.item(), xyxy) for obj, conf, xyxy in zip(object_names, detected_confidences, detected_xyxy) if conf.item() >= threshold]

        if filtered_objects:
            detected_anything = True
            for obj, _, xyxy in filtered_objects:
                x_min, _, x_max, _ = xyxy.tolist()

                if x_max < left_boundary:
                    object_positions["left"].append(obj)
                elif x_min > right_boundary:
                    object_positions["right"].append(obj)
                else:
                    object_positions["center"].append(obj)

    return object_positions, detected_anything

import sounddevice as sd
import soundfile as sf

def generate_audio_feedback(object_positions):
    """Generate and play spoken feedback."""
    environment_description = []

    for position, objs in object_positions.items():
        if objs:
            obj_counts = {obj: objs.count(obj) for obj in set(objs)}
            obj_list = [f"{count} {obj}{'s' if count > 1 else ''}" for obj, count in obj_counts.items()]
            environment_description.append(f"{', '.join(obj_list)} on the {position}.")

    if not environment_description:
        environment_description.append("No significant objects detected.")

    full_description = " ".join(environment_description)
    
    # Convert text to speech
    tts = gTTS(text=full_description, lang='en')
    audio_file = "environment_description.wav"
    tts.save(audio_file)

    # Load and play audio
    data, samplerate = sf.read(audio_file)
    sd.play(data, samplerate)
    sd.wait()  # Wait until audio finishes


def detect_objects_realtime():
    """Runs real-time object detection through a webcam."""
    cap = cv2.VideoCapture(0)  # Open webcam

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_height, frame_width, _ = frame.shape

        # Perform object detection
        results = model.predict(frame)

        # Classify object positions
        object_positions, detected_anything = classify_object_positions(results, frame_width)

        # Draw bounding boxes & labels
        for result in results:
            for box in result.boxes:
                x_min, y_min, x_max, y_max = map(int, box.xyxy[0])
                class_id = int(box.cls.item())
                label = result.names[class_id]
                confidence = box.conf.item()

                if confidence > 0.5:
                    color = (0, 255, 0)
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
                    cv2.putText(frame, f"{label} {confidence:.2f}", (x_min, y_min - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Display the frame
        cv2.imshow("Real-Time Object Detection", frame)

        # Generate and play audio feedback
        if detected_anything:
            generate_audio_feedback(object_positions)

        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Run real-time detectionq
detect_objects_realtime()
