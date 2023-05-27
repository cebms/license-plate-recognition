from ultralytics import YOLO
import supervision as sv

class PlateDetector:
    def __init__(self, weights):
        self.model = YOLO(weights)
    
    def detect_plates(self, src_image):
       result = self.model(src_image)[0]
       detections = sv.Detections.from_yolov8(result)

       #detections contains: xyxy, mask, confidence, class_id, tracker_id
       objects = [
        {
            "xyxy": xyxy,
            "confidence": confidence,
            "class_id": class_id,
        }
        for xyxy, _, confidence, class_id, _ in detections
       ]
       
       return objects

