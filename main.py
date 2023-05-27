import cv2
import argparse
from detect import PlateDetector


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, help="source image to detect")
    parser.add_argument("--weights", type=str, default="yolov8_carplate.pt", help="path to .pt weights file")

    args = parser.parse_args()
   
    image = cv2.imread(args.source)

    detector = PlateDetector(args.weights)
    plates = detector.detect_plates(image)    

    for plate in plates:
        cv2.rectangle(image, plate[0], plate[1], (0, 255, 0), 2)

    cv2.imshow("frame", image)
    cv2.waitKey(0)
