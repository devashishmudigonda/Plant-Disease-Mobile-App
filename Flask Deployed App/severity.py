import cv2
import numpy as np
from matplotlib import pyplot as plt

def show_image(title, image):
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def calculate_severity(image_path):
    original_image = cv2.imread(image_path)
    # show_image('Original Image', original_image)
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    # show_image('Grayscale Image', gray_image)
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    # show_image('Blurred Image (Noise Reduction)', blurred_image)
    _, binary_image = cv2.threshold(blurred_image, 127, 255, cv2.THRESH_BINARY_INV)
    # show_image('Binary Thresholded Image', binary_image)
    total_pixels = binary_image.size
    diseased_pixels = np.count_nonzero(binary_image)
    severity_percentage = (diseased_pixels / total_pixels) * 100
    print("\n=== Severity Calculation ===")
    print(f"Total Pixels: {total_pixels}")
    print(f"Diseased Pixels: {diseased_pixels}")
    print(f"Severity Percentage: {severity_percentage:.2f}%")
    titles = ['Original Image', 'Grayscale Image', 'Blurred Image', 'Binary Image']
    images = [original_image, gray_image, blurred_image, binary_image]
    plt.figure(figsize=(10, 10))
    for i in range(len(images)):
        plt.subplot(2, 2, i+1)
        plt.title(titles[i])
        plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
        plt.axis('off')
    plt.show()
image_path = '/Users/devashishmudigonda/Desktop/PROJECT/Plant-Disease-Prediction-main/test_images/Apple_ceder_apple_rust.JPG'
calculate_severity(image_path)