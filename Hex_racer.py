from PIL import Image
import pytesseract
import numpy as np
import os
import pyscreenshot
import keyboard
import time
TEST = False
pytesseract.pytesseract.tesseract_cmd = r'D:\Program Files\Tesseract-OCR\tesseract.exe'
os.environ['TESSDATA_PREFIX'] = r'D:\Program Files\Tesseract-OCR\tessdata'
IMG_BOX = (1050, 640, 450, 120)
def extract_yellow(image):
    # Convert the image to a NumPy array
    img_array = np.array(image)

    # Define the yellow color range in RGB
    lower_yellow = np.array([150, 150, 0], dtype=np.uint8)
    upper_yellow = np.array([255, 255, 100], dtype=np.uint8)

    # Create a mask for yellow pixels
    yellow_mask = np.all(np.logical_and(img_array[:, :, :3] >= lower_yellow, img_array[:, :, :3] <= upper_yellow), axis=-1)

    # If the image has an alpha channel, consider it as well
    if img_array.shape[2] == 4:
        yellow_mask = yellow_mask & (img_array[:, :, 3] > 0)

    # Create a new array with only the yellow part (RGB)
    yellow_part_rgb = np.zeros_like(img_array[:, :, :3])
    yellow_part_rgb[yellow_mask] = [255, 255, 0]  # Yellow color in RGB

    # Create a new array with alpha channel
    alpha_channel = np.ones((img_array.shape[0], img_array.shape[1]), dtype=np.uint8) * 255

    # Create a new array with yellow part and alpha channel
    yellow_part_with_alpha = np.dstack((yellow_part_rgb, alpha_channel))

    # Extract the yellow part as a PIL Image with RGBA mode
    yellow_image = Image.fromarray(yellow_part_with_alpha, 'RGBA')
    yellow_image.save("yellow_extracted.png")

    return yellow_image
def crop_with_margin(image, margin=5, output_path='cropped_with_margin.png'):

    # Convert the image to a NumPy array
    img_array = np.array(image)

    # Define the yellow color range in RGB
    lower_yellow = np.array([150, 150, 0], dtype=np.uint8)
    upper_yellow = np.array([255, 255, 100], dtype=np.uint8)

    # Check if the image has an alpha channel
    has_alpha = img_array.shape[-1] == 4

    # Create a mask for yellow pixels
    yellow_mask = np.all(np.logical_and(img_array[:, :, :3] >= lower_yellow, img_array[:, :, :3] <= upper_yellow), axis=-1)

    # If the image has an alpha channel, consider it as well
    if has_alpha:
        yellow_mask = yellow_mask & (img_array[:, :, 3] > 0)

    # Find row and column indices where the values are yellow
    rows, cols = np.where(yellow_mask)

    # Get the bounding box coordinates
    min_row, max_row = np.min(rows), np.max(rows)
    min_col, max_col = np.min(cols), np.max(cols)

    # Add margin to the bounding box
    min_row -= margin
    max_row += margin
    min_col -= margin
    max_col += margin

    # Ensure the coordinates are within valid ranges
    min_row = max(0, min_row)
    max_row = min(img_array.shape[0] - 1, max_row)
    min_col = max(0, min_col)
    max_col = min(img_array.shape[1] - 1, max_col)

    # Crop the image to the modified bounding box
    cropped_image = img_array[min_row:max_row+1, min_col:max_col+1]

    # Convert NumPy array back to PIL Image
    cropped_image_pil = Image.fromarray(cropped_image)

    # Save the cropped image
    cropped_image_pil.save(output_path)

    return cropped_image_pil
def extract_number_from_screenshot(image):
    custom_config = r'--psm 8 -c tessedit_char_whitelist=0123456789ABCDEF --tessdata-dir "D:\Program Files\Tesseract-OCR\tessdata"'
    text = pytesseract.image_to_string(image, config=custom_config, lang='better').strip()
    print(text)
    if TEST:
        exit()
    text = text[:2]
    # exit()  8
    # numbers = [int(s) for s in text.split() if s.isdigit()]
    # valid_numbers = [num for num in numbers if 0 <= num <= 255]
    return text

def capture_screenshot_pyscreenshot(region):
    """
    Capture a screenshot of the specified region using pyscreenshot.

    Parameters:
    - region: Tuple (left, top, width, height) representing the region to capture.

    Returns:
    - PIL Image: Captured screenshot.
    """
    left, top, width, height = region
    screenshot = pyscreenshot.grab(bbox=(left, top, left + width, top + height))
    screenshot.save("base_image.png")
    return screenshot

def get_number():
    if TEST:
        img = Image.open("base_image.png")
    else:
        img = capture_screenshot_pyscreenshot(IMG_BOX)

    extracted_yellow = extract_yellow(img)
    readble = crop_with_margin(extracted_yellow)
    return extract_number_from_screenshot(readble)


def press_keys_for_binary(binary_number):
    for i, bit in enumerate(binary_number, start=1):
        if bit == '1':
            keyboard.press_and_release(str(i))
            # time.sleep(0.1)
if __name__ == '__main__':
    # Replace 'your_screenshot.png' with the path to your screenshot image
    # image_path = 'base_image.png'

    while True:
        if keyboard.is_pressed('q'):
            print("Exiting the loop as 'q' is pressed.")
            exit()

        result = get_number()
        if result:
            result = int(result, 16)
            print(result)
            binary = format(result, '08b')
            press_keys_for_binary(binary)
            # time.sleep(0.5)
            keyboard.press_and_release('space')
            time.sleep(0.5)

        else:
            print(result)
            print("No valid numbers found in the screenshot.")
            exit()

