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

    recognized_plates = []
    for crop in crops:
        crop_prep = recognizer.preprocess(crop)
        cv2.imshow("plate crop", crop_prep)
        contours = recognizer.get_contours(crop_prep)
        count = 0

        license_plate = ""
        for contour in contours:
            count += 1
            x, y, w, h = cv2.boundingRect(contour)
            if (h >= crop.shape[0] // 2) and (h < int(crop.shape[0] * 0.9)):
                ch = crop_prep[y-2:y+h+2, x-2:x+w+2]
                ch_inv = cv2.bitwise_not(ch)
                cv2.imshow(f'character {count}', ch_inv)
                rec_character = recognizer.recognize_character(ch_inv)
                print(f"character: {rec_character}")
                license_plate += rec_character.strip()
        recognized_plates.append(license_plate)

    print(recognized_plates) 
    for i in range(len(plates)):
        image = cv2.putText(image, recognized_plates[i], (plates[i][1][0], plates[i][0][1] - 2), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("recognized", image)
    cv2.waitKey(0)
