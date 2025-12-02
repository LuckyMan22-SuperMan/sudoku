import pytesseract
import cv2
import numpy as np
import os

pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"


def extract_digit(cell, debug_dir=None, row=None, col=None):
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

    # Keep original cell size (warped grid gives reasonable resolution);
    # we'll resize the digit ROI later to 28x28 for OCR.

    # Quick pre-check: ensure the cell contains enough ink before attempting
    # heavy processing / OCR. This avoids OCR on nearly-empty cells that cause
    # false positives.
    try:
        tmp_thresh = cv2.adaptiveThreshold(
            cell_gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11, 2
        )
        ink_pixels = cv2.countNonZero(tmp_thresh)
        ink_rel = ink_pixels / float(tmp_thresh.size)
        if ink_pixels < max(150, 0.02 * tmp_thresh.size) or ink_rel < 0.01:
            # Very little ink — treat as empty
            return 0
    except Exception:
        pass

    # Improve local contrast (CLAHE) then threshold using Otsu
    try:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cell_gray = clahe.apply(cell_gray)
    except Exception:
        pass

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
    contour_count = len(contours)

    if contour_count == 0:
        # Debug: save thresh and return 0
        if debug_dir is not None and row is not None and col is not None:
            os.makedirs(debug_dir, exist_ok=True)
            cv2.imwrite(os.path.join(debug_dir, f'cell_{row}_{col}_thresh.png'), thresh)
            with open(os.path.join(debug_dir, 'ocr_results.txt'), 'a') as f:
                f.write(f'cell_{row}_{col}: NO_CONTOURS\n')
        return 0

    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)

    # Reject extremely small areas (allow smaller than before)
    if w * h < 20:
        # If debugging, still attempt OCR on whole canvas as fallback
        if debug_dir is not None and row is not None and col is not None:
            os.makedirs(debug_dir, exist_ok=True)
            cv2.imwrite(os.path.join(debug_dir, f'cell_{row}_{col}_thresh.png'), thresh)
            with open(os.path.join(debug_dir, 'ocr_results.txt'), 'a') as f:
                f.write(f'cell_{row}_{col}: SMALL_CONTOUR area={w*h}\n')
        # continue to try OCR on the whole processed canvas below as fallback

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

    # Prepare debug saving helper
    def save_debug(name, img):
        if debug_dir is not None and row is not None and col is not None:
            os.makedirs(debug_dir, exist_ok=True)
            cv2.imwrite(os.path.join(debug_dir, f'cell_{row}_{col}_{name}.png'), img)

    save_debug('pre_thresh', cell_gray)
    save_debug('thresh', thresh)
    save_debug('roi', roi)
    save_debug('roi_resized', roi_resized)
    save_debug('canvas', canvas)

    # Try multiple OCR configs as fallbacks and log attempts
    configs = [
        ("--oem 1 --psm 10 -c tessedit_char_whitelist=123456789", 'oem1-psm10'),
        ("--oem 1 --psm 8 -c tessedit_char_whitelist=123456789", 'oem1-psm8'),
        ("--oem 1 --psm 7 -c tessedit_char_whitelist=123456789", 'oem1-psm7'),
        ("--psm 10 -c tessedit_char_whitelist=123456789", 'psm10'),
        ("--psm 8 -c tessedit_char_whitelist=123456789", 'psm8'),
        ("--psm 7 -c tessedit_char_whitelist=123456789", 'psm7')
    ]

    ocr_text = ''
    ocr_used = None
    for cfg, name in configs:
        text_try = pytesseract.image_to_string(canvas, config=cfg).strip()
        text_try = text_try.replace('\n', '').replace(' ', '')
        if debug_dir is not None and row is not None and col is not None:
            with open(os.path.join(debug_dir, 'ocr_results.txt'), 'a') as f:
                f.write(f'cell_{row}_{col} [{name}]: "{text_try}"\n')
        if len(text_try) > 0 and text_try.isdigit():
            ocr_text = text_try
            ocr_used = name
            break

    # As extra fallback, try OCR on inverted canvas
    if ocr_text == '':
        inv = cv2.bitwise_not(canvas)
        save_debug('canvas_inverted', inv)
        for cfg, name in configs:
            text_try = pytesseract.image_to_string(inv, config=cfg).strip()
            text_try = text_try.replace('\n', '').replace(' ', '')
            if debug_dir is not None and row is not None and col is not None:
                with open(os.path.join(debug_dir, 'ocr_results.txt'), 'a') as f:
                    f.write(f'cell_{row}_{col} [inv-{name}]: "{text_try}"\n')
            if len(text_try) > 0 and text_try.isdigit():
                ocr_text = text_try
                ocr_used = 'inv-' + name
                break

    # If still empty, try OCR on the original cell area (uncropped) as last resort
    if ocr_text == '':
        try:
            orig_canvas = cv2.copyMakeBorder(cell_gray, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=255)
            save_debug('orig_canvas', orig_canvas)
            for cfg, name in configs:
                text_try = pytesseract.image_to_string(orig_canvas, config=cfg).strip()
                text_try = text_try.replace('\n', '').replace(' ', '')
                if debug_dir is not None and row is not None and col is not None:
                    with open(os.path.join(debug_dir, 'ocr_results.txt'), 'a') as f:
                        f.write(f'cell_{row}_{col} [orig-{name}]: "{text_try}"\n')
                if len(text_try) > 0 and text_try.isdigit():
                    ocr_text = text_try
                    ocr_used = 'orig-' + name
                    break
        except Exception:
            pass

    # Final logging of result
    if debug_dir is not None and row is not None and col is not None:
        with open(os.path.join(debug_dir, 'ocr_results.txt'), 'a') as f:
            f.write(f'cell_{row}_{col}: FINAL "{ocr_text}" used={ocr_used}\n')

    if len(ocr_text) == 0:
        return 0
    if ocr_text.isdigit():
        num = int(ocr_text)
        return num if 1 <= num <= 9 else 0
    return 0

def recognize_digits(digits_grid, debug_dir=None):
    recognized_grid = []

    # Clear previous debug results if requested
    if debug_dir is not None:
        import os
        if os.path.exists(debug_dir):
            # remove old files
            for fname in os.listdir(debug_dir):
                path = os.path.join(debug_dir, fname)
                try:
                    if os.path.isfile(path):
                        os.remove(path)
                except Exception:
                    pass

    for row_idx, row in enumerate(digits_grid):
        recognized_row = []
        for col_idx, cell in enumerate(row):
            if cell is not None:
                digit = extract_digit(cell, debug_dir=debug_dir, row=row_idx, col=col_idx)
                recognized_row.append(digit)
            else:
                # Optionally record empty cells in debug log
                if debug_dir is not None:
                    import os
                    os.makedirs(debug_dir, exist_ok=True)
                    with open(os.path.join(debug_dir, 'ocr_results.txt'), 'a') as f:
                        f.write(f'cell_{row_idx}_{col_idx}: EMPTY\n')
                recognized_row.append(0)
        recognized_grid.append(recognized_row)

    return recognized_grid
