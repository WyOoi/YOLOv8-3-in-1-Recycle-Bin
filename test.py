from ultralytics import YOLO
from ultralytics.yolo.v8.detect.predict import DetectionPredictor
import cv2

model = YOLO("best.pt")

while True:
  cap = cv2.VideoCapture(0)
  ret, frame = cap.read()
  cap.release()
  results = model.predict(frame, show=True)

  # Check if the user wants to exit
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

  # Print the results (optional)
  print(results)

print("Exiting object detection...")
