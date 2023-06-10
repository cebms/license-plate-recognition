import cv2
import argparse
from detect import PlateDetector
from recognize import PlateRecognizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, help="source image to detect")
    parser.add_argument("--weights", type=str, default="yolov8_carplate.pt", help="path to .pt weights file")

    args = parser.parse_args()
   
    image = cv2.imread(args.source)

    cv2.imshow("original image", image)

    detector = PlateDetector(args.weights)
    plates = detector.detect_plates(image)
    print(plates)
    crops = detector.get_crops_from_coord(image, plates)

    recognizer = PlateRecognizer()
    decoded = recognizer.recognize_plate(crops)
    
    output = recognizer.generate_detection_boxes(image, plates, decoded, (0,255,0), (0,0,255))
    cv2.imshow("output", output)
    print(decoded)
    cv2.waitKey(0)
