from ultralytics import YOLO
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, help="source image to detect")
    parser.add_argument("--weights", type=str, default="yolov8n.pt", help="path to .pt weights file")

    args = parser.parse_args()

    model = YOLO(args.weights)
    model.predict(source=args.source, show=True)

