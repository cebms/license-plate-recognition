import pytesseract
import cv2

class PlateRecognizer:
    def __init__(self):
        pass

    def preprocess(self, image):
        #Apply grayscale to the image
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #Apply gaussian blur
        gaussian_blurred = cv2.GaussianBlur(grayscale, (5,5), 0)
        #Apply Otsu's thresholding
        value, otsu = cv2.threshold(gaussian_blurred, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

        return otsu 

