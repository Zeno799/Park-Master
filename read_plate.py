import cv2
import pytesseract
import numpy as np

# Configure pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def read_license_plate(image_path):
    # Load the image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # Ensure the image is loaded
    if img is None:
        print("Error: Image not loaded correctly.")
        return

    # Apply some pre-processing to the image
    # Apply Gaussian blur to remove noise
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    
    # Apply thresholding to get a binary image
    _, binary_image = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Optionally show the binary image
    # cv2.imshow("Binary Image", binary_image)
    # cv2.waitKey(0)

    # Use Tesseract to extract text from the binary image
    custom_config = r'--oem 3 --psm 6'
    plate_text = pytesseract.image_to_string(binary_image, config=custom_config)
    
    # Clean up the plate text
    plate_text = ''.join(e for e in plate_text if e.isalnum())
    
    print(f"Detected License Plate Text: {plate_text}")

    # Display the original and processed image with detected text
    cv2.imshow("Original Image", img)
    cv2.imshow("Processed Image", binary_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return plate_text

# Path to your license plate image
image_path = './test_images/binary_image.jpg'

# Call the function
read_license_plate(image_path)
