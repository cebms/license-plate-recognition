from detect import PlateDetector
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, help="source image to detect")
    parser.add_argument("--weights", type=str, default="yolov8_carplate.pt", help="path to .pt weights file")

    args = parser.parse_args()
    
    detector = PlateDetector(args.weights)

    objects = detector.detect_plates(args.source)    

    print(objects)
