from ultralytics import YOLO
import supervision as sv
import cv2

class PlateDetector:
    def __init__(self, weights):
        self.model = YOLO(weights)
    
    def detect_plates(self, src_image):
       result = self.model(src_image)[0]
       detections = sv.Detections.from_yolov8(result)

       #detections contains: xyxy, mask, confidence, class_id, tracker_id
       plates_coordinates = [
        [(int(xyxy[0]), int(xyxy[1])),(int(xyxy[2]), int(xyxy[3]))]
        for xyxy, _, _, class_id, _ in detections
        if class_id == 0
       ]

       
       return plates_coordinates 


    def get_crops_from_coord(self, original_img, plates_coords):
        crops = []
        for plate in plates_coords:
           crops.append(original_img[plate[0][1]:plate[1][1], plate[0][0]:plate[1][0]])
           cv2.rectangle(original_img, plate[0], plate[1], (0, 255, 0), 2)
        return crops

