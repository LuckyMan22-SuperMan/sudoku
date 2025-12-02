import sys
import cv2
from image_processing import preprocess_image, extract_grid, extract_digits
from digit_recognition import recognize_digits
from solver import solve_sudoku, is_board_valid

def display_board(board):
    for row in board:
        print(" ".join(str(num) for num in row))

def main(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image not found.")
        return

    gray, processed = preprocess_image(image)

    # Warp the original grayscale image using grid contours detected on `processed`
    grid = extract_grid(processed, original_image=gray)
    if grid is None:
        print("Could not detect Sudoku grid.")
        return

    digits = extract_digits(grid)
    board = recognize_digits(digits)

    print("Initial Sudoku board detected:")
    display_board(board)

    if not is_board_valid(board):
        print("Detected invalid Sudoku board (contradictions). Cannot solve.")
        return

    if solve_sudoku(board):
        print("\nSudoku solved:")
        display_board(board)
    else:
        print("No solution exists.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python main.py path_to_sudoku_image")
    else:
        main(sys.argv[1])
