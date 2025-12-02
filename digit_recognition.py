import pytesseract
import cv2
import numpy as np

pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"


def extract_digit(cell):
    """Return recognized digit (1-9) from a cell image or 0 if empty/unknown.

    The function:
    - Accepts a grayscale or binary cell image.
    - Thresholds with Otsu and forces a white background with black digit.
    - Keeps the largest contour (likely the digit), crops, centers and pads it to a 28x28 canvas.
    - Adds a white border and runs Tesseract with single-character PSM.
    """
    if cell is None:
        return 0

    # Ensure grayscale
    if len(cell.shape) == 3:
        cell_gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
    else:
        cell_gray = cell.copy()

    # Resize to a working size (keep some detail)
    cell_gray = cv2.resize(cell_gray, (28, 28), interpolation=cv2.INTER_AREA)

    # Threshold using Otsu to get a clean binary image
    _, thresh = cv2.threshold(cell_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Ensure white background and black digit (Tesseract expects dark text on light background)
    white_pixels = cv2.countNonZero(thresh)
    black_pixels = thresh.size - white_pixels
    if white_pixels < black_pixels:
        thresh = cv2.bitwise_not(thresh)

    # Remove small noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # Find contours - keep largest contour as digit
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0

    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)

    # Reject extremely small areas
    if w * h < 30:
        return 0

    roi = thresh[y:y + h, x:x + w]

    # Resize ROI to fit in 20x20 box while preserving aspect ratio, then center in 28x28
    h_r, w_r = roi.shape
    scale = 20.0 / max(h_r, w_r)
    new_w = max(1, int(w_r * scale))
    new_h = max(1, int(h_r * scale))
    roi_resized = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = 255 * np.ones((28, 28), dtype=np.uint8)
    x_offset = (28 - new_w) // 2
    y_offset = (28 - new_h) // 2
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = roi_resized

    # Add white border to give Tesseract some margin
    canvas = cv2.copyMakeBorder(canvas, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=255)

    # OCR config: single character digits 1-9
    config = "--psm 10 -c tessedit_char_whitelist=123456789"

    text = pytesseract.image_to_string(canvas, config=config).strip()
    text = text.replace('\n', '').replace(' ', '')

    if len(text) == 0:
        return 0
    if text.isdigit():
        num = int(text)
        return num if 1 <= num <= 9 else 0
    return 0

def recognize_digits(digits_grid):
    recognized_grid = []

    for row_idx, row in enumerate(digits_grid):
        recognized_row = []
        for col_idx, cell in enumerate(row):
            if cell is not None:
                digit = extract_digit(cell)
                recognized_row.append(digit)
            else:
                recognized_row.append(0)
        recognized_grid.append(recognized_row)

    return recognized_grid
