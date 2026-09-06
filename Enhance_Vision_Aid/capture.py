import cv2
import os
import time

def capture_and_save_frame():
    """Waits for 3 seconds before capturing a frame, then saves it."""
    cap = cv2.VideoCapture(0)  # Open webcam

    if not cap.isOpened():
        print("Error: Unable to access the webcam.")
        return
    
     # Wait for 3 seconds before capturing

  # Capture multiple frames to allow auto-adjustment
    ret, frame = cap.read()
    print("Adjusting camera settings... Please wait 3 seconds.")
    time.sleep(1)
    ret, frame = cap.read()

    if ret:
        save_path = os.path.join("F:\Enhanced_Vision_Aid", "captured_frame.jpg")  # Save in the current directory
        cv2.imwrite(save_path, frame)  # Save the image
        print(f"Frame captured and saved at: {save_path}")

        cap.release()  # Release webcam
        cv2.destroyAllWindows()
        
        # Pass the image to context_aware()
        context_aware(save_path)  
    else:
        print("Error: Failed to capture frame.")
        cap.release()
        cv2.destroyAllWindows()

def context_aware(image_path):
    """Loads and displays the saved image."""
    print(f"🔍 Processing image: {image_path}")
    frame = cv2.imread(image_path)  # Read the saved image
    if frame is None:
        print("Error: Image not found or failed to load.")
        return

    cv2.imshow("Captured Frame", frame)
    cv2.waitKey(2000)  # Show for 2 seconds
    cv2.destroyAllWindows()

# Run the function to capture and process the frame
capture_and_save_frame()
