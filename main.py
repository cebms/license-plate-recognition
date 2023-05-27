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

    crops = []
    for plate in plates:
        crops.append(image[plate[0][1]:plate[1][1], plate[0][0]:plate[1][0]])
        cv2.rectangle(image, plate[0], plate[1], (0, 255, 0), 2)
    
    recognizer = PlateRecognizer()

    for crop in crops:
        crop_prep = recognizer.preprocess(crop)
        cv2.imshow("plate crop", crop_prep)
        contours = recognizer.get_contours(crop_prep)

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            crop_contours = cv2.rectangle(crop, (x, y), (x+w, y+h), (0,0,255), 2)
            cv2.imshow("contours", crop_contours)
        # cv2.imshow("contours", with_contours)

    cv2.waitKey(0)
