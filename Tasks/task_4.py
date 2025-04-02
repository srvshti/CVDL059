import cv2
import torch
import time
from ultralytics import YOLO

# Enable cuDNN benchmark for potential speed improvements
torch.backends.cudnn.benchmark = True

# Determine device: use GPU if available, otherwise CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load YOLOv8 model and move it to the appropriate device
model = YOLO('yolov8n.pt')
model.to(device)

# Optionally convert the model to half precision for faster inference (if supported)
if device == 'cuda':
    model.half()

# Class names for COCO dataset (YOLO default classes)
CLASS_NAMES = model.names

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

target_fps = 15
target_interval = 1.0 / target_fps
prev_time = time.time()

while True:
    loop_start_time = time.time()
    
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Optionally resize the frame to lower resolution for faster processing
    frame_resized = cv2.resize(frame, (640, 480))

    # Convert frame to appropriate type for FP16 if using half precision
    if device == 'cuda':
        frame_input = frame_resized.astype('float32')
        # Normalize or preprocess if needed, then convert to half precision:
        # Here, we assume the model handles preprocessing internally.
        frame_input = torch.from_numpy(frame_input).permute(2, 0, 1).unsqueeze(0).to(device).half()
        results = model(frame_input)
    else:
        results = model(frame_resized)

    detections_list = []
    # Processing detections remains similar
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)

        for box, conf, class_id in zip(boxes, confidences, class_ids):
            x1, y1, x2, y2 = map(int, box)
            class_name = CLASS_NAMES.get(class_id, "Unknown")
            detection = {
                "class": class_name,
                "confidence": round(float(conf), 2),
                "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
            }
            detections_list.append(detection)
            label = f"{class_name} | {conf:.2f}"
            cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame_resized, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Calculate and display FPS on the resized frame
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time
    cv2.putText(frame_resized, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    cv2.imshow("YOLOv8 Live Feed", frame_resized)
    print(detections_list)

    processing_time = time.time() - loop_start_time
    delay = max(1, int((target_interval - processing_time) * 1000))
    if cv2.waitKey(delay) == 27:
        break

cap.release()
cv2.destroyAllWindows()
