import pytesseract
import cv2

class PlateRecognizer:
    def __init__(self):
        self.char_recog_tess_config = "-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 8 --oem 3"
        pass

    def preprocess(self, image):
        self.original_image = image.copy()
        #Apply grayscale to the image
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # cv2.imshow("gs1", grayscale)
        grayscale = cv2.resize(grayscale,
                               None,
                               fx = 3,
                               fy = 3,
                               interpolation = cv2.INTER_CUBIC
                               )
        # cv2.imshow("gs2", grayscale)
        #Apply gaussian blur
        gaussian_blurred = cv2.GaussianBlur(grayscale, (5,5), 0)
        # cv2.imshow("blurred", gaussian_blurred)
        #Apply Otsu's thresholding
        _ , otsu = cv2.threshold(gaussian_blurred, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        # cv2.imshow("otsu", otsu)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,  (3,3))
        otsu = cv2.dilate(otsu, kernel)

        # cv2.imshow("dilated", otsu)
        
        return otsu

    def get_contours(self, image):
        #Find contours considering only external elements
        contours, _ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        #Sorts contours relative to the x value of their bounding rectangles
        sorted_contours = sorted(contours, key = lambda contour: cv2.boundingRect(contour)[0])

        return sorted_contours

    def recognize_plate(self, plate_crops):
        recognized_plates = []
        for crop in plate_crops:
            original_crop_resized = cv2.resize(crop, None, fx = 3, fy = 3, interpolation =  cv2.INTER_CUBIC)
            crop_prep = self.preprocess(crop)
            cv2.imshow("plate crop", crop_prep)
            contours = self.get_contours(crop_prep)

            license_plate = ""
            count = 0
            for contour in contours: 
                x, y, w, h = cv2.boundingRect(contour)
                if (h >= original_crop_resized.shape[0] // 2) and (h < int(original_crop_resized.shape[0] * 0.75)):
                    print(f"shape: {original_crop_resized.shape[0], original_crop_resized.shape[1]}")
                    ch = crop_prep[y-int(original_crop_resized.shape[0]*0.04):y+h+int(original_crop_resized.shape[0]*0.04),
                                   x-int(original_crop_resized.shape[1]*0.01):x+w+int(original_crop_resized.shape[1]*0.01)
                                  ]
                    ch_inv = cv2.bitwise_not(ch)
                    count += 1
                    ch_inv = cv2.medianBlur(ch_inv, 5)
                    cv2.imshow(f'character {count}', ch_inv)
                    rec_character = self.recognize_character(ch_inv)
                    print(f"character: {rec_character}")
                    license_plate += rec_character.strip()
            recognized_plates.append(license_plate)

        return recognized_plates


    def recognize_character(self, image):
       character = pytesseract.image_to_string(image, config=self.char_recog_tess_config) 
       return character


    def generate_detection_boxes(self, original_image, plates_coords, detection_strings, bb_color, text_color):
       final_img = original_image.copy()
       if len(plates_coords) == len(detection_strings):
           for i in range(len(plates_coords)):
               final_img = cv2.rectangle(final_img, plates_coords[i][0], plates_coords[i][1], bb_color, 2)
               final_img = cv2.putText(final_img,
                                       detection_strings[i],
                                       (plates_coords[i][0][0], plates_coords[i][0][1]),
                                       cv2.FONT_HERSHEY_COMPLEX,
                                       1,
                                       text_color,
                                       2)

       return final_img




