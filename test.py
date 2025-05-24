from ultralytics import YOLO

# Load the model and run inference on the webcam
YOLO('D:\\1. Semester 7\\LaskarAI\\restro\\fix\\runs\\pose\\train3\\weights\\best.pt').predict(source=0, show=True, save=True, project='runs/pose/output', name='tests')