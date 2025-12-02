import cv2
import numpy as np

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left

    return rect

def preprocess_image(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Adaptive thresholding (inverted for contour detection)
    thresh = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11, 2
    )
    # Return both grayscale (for OCR/wrapping) and processed (for contour detection)
    return gray, thresh

def extract_grid(processed_image, original_image=None):
    """Find the largest 4-point contour in `processed_image` and warp the `original_image` if provided.

    Returns the warped grayscale grid (square) suitable for per-cell OCR, or None if not found.
    """
    contours, _ = cv2.findContours(
        processed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for contour in contours:
        approx = cv2.approxPolyDP(
            contour, 0.02 * cv2.arcLength(contour, True), True
        )
        if len(approx) == 4:
            pts = np.array([point[0] for point in approx], dtype='float32')
            pts = order_points(pts)

            widthA = np.linalg.norm(pts[2] - pts[3])  # bottom width
            widthB = np.linalg.norm(pts[1] - pts[0])  # top width
            maxWidth = max(int(widthA), int(widthB))

            heightA = np.linalg.norm(pts[1] - pts[2])  # right height
            heightB = np.linalg.norm(pts[0] - pts[3])  # left height
            maxHeight = max(int(heightA), int(heightB))

            side = max(maxWidth, maxHeight)
            side = max(side, 450)  # minimum size for consistency

            dst = np.array([[0, 0], [side - 1, 0], [side - 1, side - 1], [0, side - 1]], dtype='float32')
            matrix = cv2.getPerspectiveTransform(pts, dst)
            # Prefer to warp the original grayscale for OCR; fallback to processed_image
            if original_image is not None:
                warped = cv2.warpPerspective(original_image, matrix, (int(side), int(side)))
            else:
                warped = cv2.warpPerspective(processed_image, matrix, (int(side), int(side)))
            return warped

    print("No grid detected.")
    return None

def extract_digits(grid_gray):
    """Split the square `grid_gray` into 9x9 cells and return list of cell images or None.

    `grid_gray` should be a grayscale image (warped original). Detection uses local adaptive
    thresholding on each cell to decide if a digit is present.
    """
    cell_size = grid_gray.shape[0] // 9
    digits = []

    for row in range(9):
        digits_row = []
        for col in range(9):
            x, y = col * cell_size, row * cell_size
            cell = grid_gray[y:y + cell_size, x:x + cell_size]
            # locally threshold cell to detect ink
            cell_thresh = cv2.adaptiveThreshold(
                cell, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                11, 2
            )
            nonzero = cv2.countNonZero(cell_thresh)
            # If a reasonable fraction of pixels are non-zero, assume a digit exists
            if nonzero > (cell_size * cell_size * 0.01):  # >1% area
                digits_row.append(cell)
            else:
                digits_row.append(None)
        digits.append(digits_row)

    return digits
