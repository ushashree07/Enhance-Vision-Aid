import cv2
import torch
import speech_recognition as sr
from ultralytics import YOLO
from gtts import gTTS
import sounddevice as sd
import soundfile as sf

# Load YOLOv10 Model
model = YOLO("weights/yolov10s.pt")

# Voice recognition function
def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening for command...")
        recognizer.adjust_for_ambient_noise(source)
        try:
            audio = recognizer.listen(source, timeout=5)
            command = recognizer.recognize_google(audio).lower()
            print(f"Recognized command: {command}")
            return command
        except sr.UnknownValueError:
            return "error"
        except sr.RequestError:
            return "error"

# Object position classification
def classify_object_positions(results, frame_width):
    object_positions = {"left": [], "center": [], "right": []}
    detected_anything = False

    left_boundary = frame_width * 0.33
    right_boundary = frame_width * 0.66

    for result in results:
        detected_boxes = result.boxes
        detected_classes = detected_boxes.cls
        detected_confidences = detected_boxes.conf
        detected_xyxy = detected_boxes.xyxy

        object_names = [result.names[int(cls.item())] for cls in detected_classes]
        threshold = 0.5
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

# Function to generate and play audio feedback

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

# Main function for real-time detection with voice assistant
def detect_objects_realtime():
    """Voice-controlled real-time object detection."""
    cap = cv2.VideoCapture(0)  # Open webcam
    is_active = False  # Flag to track detection status

    while True:
        print("\nSay 'start detection' to begin, 'stop detection' to exit, or 'exit' to quit.")

        command = recognize_speech()
        if "start detection" in command:
            is_active = True
            print("Object detection activated. Say 'stop detection' to stop.")

        elif "stop detection" in command:
            is_active = False
            print("Object detection stopped.")

        elif "exit" in command:
            print("Exiting program.")
            break

        while is_active:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame.")
                break
            
            frame_height, frame_width, _ = frame.shape
            results = model.predict(frame)
            object_positions, detected_anything = classify_object_positions(results, frame_width)

            if detected_anything:
                generate_audio_feedback(object_positions)
            else:
                print("No significant objects detected.")

            # Check for stop command in between frames
            print("Say 'stop detection' to exit detection mode.")
            command = recognize_speech()
            if "stop detection" in command or "exit" in command:
                is_active = False
                print("Object detection stopped.")

    cap.release()
    cv2.destroyAllWindows()

# Run the voice-controlled object detection
detect_objects_realtime()