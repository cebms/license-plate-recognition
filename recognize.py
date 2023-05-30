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
        #Apply gaussian blur
        gaussian_blurred = cv2.GaussianBlur(grayscale, (5,5), 0)
        #Apply Otsu's thresholding
        value, otsu = cv2.threshold(gaussian_blurred, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        
        return otsu

    def get_contours(self, image):
        #Find contours considering only external elements
        contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        return contours

    def recognize_character(self, image):
       character = pytesseract.image_to_string(image, config=self.char_recog_tess_config) 
       return character



