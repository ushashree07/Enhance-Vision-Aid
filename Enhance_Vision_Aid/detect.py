from ultralytics import YOLO
import os
import cv2
import supervision as sv
#from IPython.display import display, Image

# Load the YOLO model
model = YOLO('F:/Enhanced_Vision_Aid/weights/yolov10s.pt')
model2=YOLO('F:\Enhanced_Vision_Aid\weights\yolov10b.pt')
# Run predictions on an image and save the results
#results = model.predict('F:/Enhanced_Vision_Aid/Testing-images/im11.jpeg', save=True)
#results2 = model2.predict('F:/Enhanced_Vision_Aid/Testing-images/im11.jpeg', save=True)


cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model2(frame)[0]
    detections = sv.Detections.from_ultralytics(results)

    bounding_box_annotator = sv.BoundingBoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    annotated_image = bounding_box_annotator.annotate(
        scene=frame, detections=detections)
    annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)
    cv2.imshow('YOLO', annotated_image)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
