import streamlit as st
from PIL import Image
import cv2
import numpy as np
import os
import base64
from io import BytesIO
from streamlit_paste_button import paste_image_button as pbutton

# --- App Configuration ---
st.set_page_config(page_title="Gap Challenge Solver", layout="wide")

# --- Constants ---
CONFIDENCE_THRESHOLD_SHAPE = 0.60
BLANK_STD_DEV_THRESHOLD = 15.0    
TEMPLATE_DIR = "templates"
WHITE_THRESHOLD = 240

# --- Helper Functions ---
def find_empty(board):
    for r in range(len(board)):
        for c in range(len(board[0])):
            if board[r][c] == 'blank':
                return (r, c)
    return None

def solve_with_backtracking(board, all_shapes):
    find = find_empty(board)
    if not find: return True
    row, col = find
    for shape in all_shapes:
        if shape in board[row] or shape in [board[i][col] for i in range(len(board))]: continue
        board[row][col] = shape
        if solve_with_backtracking(board, all_shapes): return True
        board[row][col] = 'blank'
    return False

def find_question_mark_solution(board, universe_of_shapes):
    question_pos = None
    for r in range(len(board)):
        for c in range(len(board[0])):
            if board[r][c] == '6question':
                question_pos = (r, c)
                break
        if question_pos: break
    if not question_pos:
        st.error("No '6question' mark found on the board.")
        return None, None
    board_copy = [row[:] for row in board]
    qr, qc = question_pos
    board_copy[qr][qc] = 'blank'
    if solve_with_backtracking(board_copy, universe_of_shapes):
        solution_shape = board_copy[qr][qc]
        return solution_shape, board_copy
    else:
        return None, None

def crop_to_grid(source_image: np.ndarray):
    gray = cv2.cvtColor(source_image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, WHITE_THRESHOLD, 255, cv2.THRESH_BINARY_INV)
    non_zero_pixels = cv2.findNonZero(thresh)
    if non_zero_pixels is None: return source_image
    x, y, w, h = cv2.boundingRect(non_zero_pixels)
    return source_image[y:y+h, x:x+w]

def recognize_shape_in_cell(cell_roi_color, templates):
    cell_roi_gray = cv2.cvtColor(cell_roi_color, cv2.COLOR_BGR2GRAY)
    std_dev = np.std(cell_roi_gray)
    if std_dev < BLANK_STD_DEV_THRESHOLD: return "blank"
    best_match = {'label': 'blank', 'score': -1.0}
    for label, template_with_alpha in templates.items():
        if template_with_alpha.shape[2] == 4:
            mask = template_with_alpha[:,:,3]
            template_color = cv2.cvtColor(template_with_alpha, cv2.COLOR_BGRA2BGR)
        else:
            template_color = template_with_alpha
            template_gray_for_mask = cv2.cvtColor(template_color, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(template_gray_for_mask, 10, 255, cv2.THRESH_BINARY)
        h, w, _ = template_color.shape
        for scale in np.linspace(0.5, 1.0, 10):
            scaled_h, scaled_w = int(h * scale), int(w * scale)
            if not (scaled_h > 0 and scaled_w > 0 and scaled_h <= cell_roi_color.shape[0] and scaled_w <= cell_roi_color.shape[1]): continue
            scaled_template = cv2.resize(template_color, (scaled_w, scaled_h))
            scaled_mask = cv2.resize(mask, (scaled_w, scaled_h))
            result = cv2.matchTemplate(cell_roi_color, scaled_template, cv2.TM_CCOEFF_NORMED, mask=scaled_mask)
            _, max_val, _, _ = cv2.minMaxLoc(result)
            if not np.isfinite(max_val): max_val = -1.0
            if max_val > best_match['score']:
                best_match.update({'score': max_val, 'label': label})
    final_label = 'blank'
    if best_match['score'] > CONFIDENCE_THRESHOLD_SHAPE:
        final_label = best_match['label']
    return final_label

def recognize_grid_and_options(image: Image.Image, grid_size: int, templates):
    original_image = np.array(image.convert('RGB'))
    original_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)
    border_size = 20
    bordered_image = cv2.copyMakeBorder(original_image, top=border_size, bottom=border_size, left=border_size, right=border_size, borderType=cv2.BORDER_CONSTANT, value=[255, 255, 255])
    cropped_color = crop_to_grid(bordered_image)
    img_h, img_w, _ = cropped_color.shape
    estimated_total_rows = grid_size + 1.1 
    estimated_cell_h = img_h / estimated_total_rows
    estimated_grid_h = int(estimated_cell_h * grid_size)
    puzzle_area = cropped_color[:estimated_grid_h, :]
    options_area = cropped_color[estimated_grid_h:, :]
    puzzle_area = crop_to_grid(puzzle_area)
    standard_size = 600
    aligned_color_image = cv2.resize(puzzle_area, (standard_size, standard_size))
    cell_height, cell_width = standard_size // grid_size, standard_size // grid_size
    output_grid = [["" for _ in range(grid_size)] for _ in range(grid_size)]
    for r in range(grid_size):
        for c in range(grid_size):
            cell_roi = aligned_color_image[r*cell_height:(r+1)*cell_height, c*cell_width:(c+1)*cell_width]
            output_grid[r][c] = recognize_shape_in_cell(cell_roi, templates)
    universe_of_shapes = []
    if options_area.shape[0] > 10:
        option_cell_h = options_area.shape[0]
        option_cell_w_est = option_cell_h 
        num_options = round(options_area.shape[1] / option_cell_w_est)
        cell_w_options = options_area.shape[1] // num_options
        for i in range(num_options):
            option_roi = options_area[:, i*cell_w_options:(i+1)*cell_w_options]
            option_roi = cv2.resize(option_roi, (cell_width, cell_height))
            shape_in_option = recognize_shape_in_cell(option_roi, templates)
            if shape_in_option not in ['blank', '6question']:
                universe_of_shapes.append(shape_in_option)
    for r in range(grid_size):
        for c in range(grid_size):
            shape_in_grid = output_grid[r][c]
            if shape_in_grid not in ['blank', '6question']:
                universe_of_shapes.append(shape_in_grid)
    if not universe_of_shapes:
        universe_of_shapes = ['1circle','2triangle','3square','4cross','5star']
    return output_grid, sorted(list(set(universe_of_shapes)))

# --- UI Layout ---
st.title("🧩 Gap Challenge Solver")
st.info("To solve, take a screenshot of the puzzle, then click the button below and press Ctrl+V (or Cmd+V).")

with st.sidebar:
    st.header("Controls")
    grid_size = st.radio("Grid Size", (4, 5), index=0)
    is_aon = st.toggle("AON Puzzle", value=True, help="For future use.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Your Puzzle")
    paste_result = pbutton("📋 Paste an image")

    if "pasted_image" not in st.session_state:
        st.session_state.pasted_image = None
    
    if paste_result.image_data is not None:
        st.session_state.pasted_image = paste_result.image_data

    if st.session_state.pasted_image:
        st.image(st.session_state.pasted_image, caption="Pasted from Clipboard", use_column_width=True)
    else:
        st.write("Awaiting a pasted image...")

with col2:
    st.subheader("Solution")
    if st.session_state.pasted_image:
        with st.spinner("Analyzing puzzle..."):
            image = st.session_state.pasted_image
            shape_labels = ['1circle','2triangle','3square','4cross','5star','6question']
            
            # Template Loading Check
            try:
                if not os.path.exists(TEMPLATE_DIR):
                     st.error(f"Directory '{TEMPLATE_DIR}' not found! Please upload it to GitHub.")
                     st.stop()
                     
                templates = {}
                for label in shape_labels:
                    path = os.path.join(TEMPLATE_DIR, f"{label}.png")
                    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                    if img is None:
                        st.error(f"Missing template file: {path}")
                        st.stop()
                    templates[label] = img
            except Exception as e:
                st.error(f"Error loading template files: {e}"); st.stop()
            
            initial_grid, detected_shapes = recognize_grid_and_options(image, grid_size, templates)
            
            if initial_grid and detected_shapes:
                st.write("Detected Initial Grid:")
                st.table(initial_grid)
                solution_shape, solved_grid = find_question_mark_solution(initial_grid, detected_shapes)
                if solution_shape and solved_grid:
                    st.success("Solution Found!")
                    st.metric(label="The shape for the '?' is", value=solution_shape)
                    st.write("Completed Grid:")
                    st.table(solved_grid)
                else:
                    st.error("Could not find a valid solution for this puzzle.")
    else:
        st.write("Click the paste button and press Ctrl+V to see the solution.")