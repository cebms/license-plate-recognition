# Car License Plate Recognizer


The Car License Plate Recognizer is a Python project that utilizes various computer vision and OCR techniques to detect and recognize car license plates from an input image. It combines the power of OpenCV, YOLO v8, and Tesseract OCR to achieve accurate and efficient results.

- [Introduction](#introduction)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Screenshots](#screenshots)
- [License](#license)

## Introduction

The Car License Plate Recognizer is designed to take an input image containing one vehicle and then proceed through a series of steps to isolate and extract license plate information. The workflow of the project includes:

1. Using YOLO v8 with a specific license plate detection model to find the bounding boxes of license plates in the input image.
2. Cropping the original image to isolate each license plate individually.
3. Preprocessing each license plate image by applying greyscale, Gaussian blur, Otsu's Thresholding, and dilation techniques.
4. Using OpenCV's `findContours` function to separate each character in the license plate image.
5. Sorting the detected contours based on their x-coordinate to ensure correct character order.
6. Passing each character image to Tesseract OCR to extract the characters' text.
7. Concatenating the OCR results to obtain the final car plate string.

## Technologies Used

The project is primarily coded in Python and relies on the following libraries and frameworks:

- OpenCV: A powerful computer vision library used for image processing, contour detection, and more;
- YOLO v8: A deep learning-based object detection algorithm used for identifying license plate bounding boxes;
- Tesseract OCR: An optical character recognition engine used to recognize characters from the preprocessed license plate images.

## Installation

To run the Car License Plate Recognizer on your local machine, follow these steps:

1. Clone this repository to your local system.
2. Install the required dependencies using `pip`:

   ```bash
   pip install opencv-python
   pip install pytesseract
   pip install ultralytics
   ```

3. Make sure you have Tesseract OCR installed on your system. You can download and install it from the [official Tesseract GitHub repository](https://github.com/tesseract-ocr/tesseract).
   If using Debian-based Linux distros, you can just use:
   ```
   sudo apt install tesseract-ocr
   ```

## Usage

To use the Car License Plate Recognizer, follow these instructions:

1. Place the input image (containing vehicles and license plates) in the project directory.
2. Run the Python script that implements the license plate recognition pipeline.
3. In order to specify the image path, make sure to use the ``` --path ``` flag followed by your image's path
4. It is not necessary to specify the YOLO weights file, but if you wish to change it for your own trained model, use the ``` --weights ``` flag

The script will process the input image, detect license plates, perform character recognition, and output the recognized license plate strings on a new window.

## Screenshots

Here are two screenshots showcasing the Car License Plate Recognizer in action:

![p1](https://github.com/cebms/license-plate-recognition/assets/59201335/501bec0b-a4d2-4124-9385-dd07fed96016)

![p2](https://github.com/cebms/license-plate-recognition/assets/59201335/5d8da774-c46f-427d-8256-c158e8c9adde)



## License

This project is licensed under the [MIT License](LICENSE).

---

We hope you find the Car License Plate Recognizer useful! If you encounter any issues or have suggestions for improvement, please feel free to open an issue or submit a pull request. Happy license plate recognition! 🚗🕵️‍♂️
