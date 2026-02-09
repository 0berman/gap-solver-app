import streamlit as st
from streamlit_paste_button import paste_image_button as pbutton
import cv2
import numpy as np
import joblib
import os
import time

# Import your robust preprocessor
# (Ensure preprocess.py is in the same folder)
from preprocess import standardize_cell

# ==========================================
# 1. PAGE & SESSION SETUP
# ==========================================
st.set_page_config(page_title="Gap Solver AI", layout="wide", page_icon="🧩")

if 'grid_data' not in st.session_state: st.session_state.grid_data = None
if 'last_paste_id' not in st.session_state: st.session_state.last_paste_id = 0

# Callback to clear state explicitly (extra safety)
def clear_grid_state():
    st.session_state.grid_data = None
    st.session_state.last_paste_id = 0

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
@st.cache_resource
def load_model():
    if os.path.exists("model.pkl"):
        return joblib.load("model.pkl")
    return None

def smart_crop_board(image, grid_n):
    img = image.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 11, 2
    )
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    expected_w = img.shape[1] // grid_n
    valid_boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if 0.7 < w/h < 1.3 and expected_w*0.2 < w < expected_w*1.5:
            valid_boxes.append((x,y,w,h))
            
    if len(valid_boxes) < 4: return img
    
    min_x = min([b[0] for b in valid_boxes])
    min_y = min([b[1] for b in valid_boxes])
    max_x = max([b[0]+b[2] for b in valid_boxes])
    max_y = max([b[1]+b[3] for b in valid_boxes])
    
    if (max_x - min_x) < img.shape[1] * 0.2: return img
    
    return img[min_y:max_y, min_x:max_x]

def predict_grid_labels(image, grid_size, clf):
    # 1. Find Board
    board = smart_crop_board(image, grid_size)
    
    h, w = board.shape[:2]
    cell_h, cell_w = h // grid_size, w // grid_size
    
    batch_features = []
    
    for r in range(grid_size):
        for c in range(grid_size):
            y1, x1 = r*cell_h, c*cell_w
            m_h = int(cell_h * 0.05)
            m_w = int(cell_w * 0.05)
            
            raw_cell = board[y1:y1+cell_h, x1:x1+cell_w]
            safe_cell = raw_cell[m_h:cell_h-m_h, m_w:cell_w-m_w]
            
            clean = standardize_cell(safe_cell)
            feat = clean.flatten() / 255.0
            batch_features.append(feat)
            
    if not batch_features: return [], board
    
    predictions = clf.predict(np.array(batch_features))
    
    grid_matrix = []
    idx = 0
    for r in range(grid_size):
        row = []
        for c in range(grid_size):
            row.append(predictions[idx])
            idx += 1
        grid_matrix.append(row)
        
    return grid_matrix, board

# ==========================================
# 3. SOLVER LOGIC
# ==========================================
def solve_backtracking(board, shapes):
    def find_empty(b):
        for r in range(len(b)):
            for c in range(len(b[0])):
                if b[r][c] == 'blank': return (r, c)
        return None

    def is_valid(b, s, pos):
        # Row check
        for i in range(len(b[0])):
            if b[pos[0]][i] == s and pos[1] != i: return False
        # Col check
        for i in range(len(b)):
            if b[i][pos[1]] == s and pos[0] != i: return False
        return True

    find = find_empty(board)
    if not find: return True
    row, col = find

    for shape in shapes:
        if is_valid(board, shape, (row, col)):
            board[row][col] = shape
            if solve_backtracking(board, shapes): return True
            board[row][col] = 'blank'
            
    return False

def run_solver_logic(grid_data):
    if not grid_data: return "No Data", None
    board_copy = [row[:] for row in grid_data]
    current_grid_size = len(board_copy)
    
    present_shapes = set()
    question_pos = None
    
    for r in range(current_grid_size):
        for c in range(current_grid_size):
            val = board_copy[r][c]
            if val == '6question':
                question_pos = (r, c)
                board_copy[r][c] = 'blank'
            elif val != 'blank':
                present_shapes.add(val)
                
    if not question_pos: return "No '?' found!", None
        
    universe = sorted(list(present_shapes))
    standard_shapes = ['1circle', '2triangle', '3square', '4cross', '5star']
    
    for s in standard_shapes:
        if len(universe) >= current_grid_size: break
        if s not in universe: universe.append(s)
    universe.sort()
    
    success = solve_backtracking(board_copy, universe)
    
    if success:
        qr, qc = question_pos
        return board_copy[qr][qc], board_copy
    return "Unsolvable", None

# ==========================================
# 4. SIDEBAR & CONFIG
# ==========================================
with st.sidebar:
    st.header("⚙️ Settings")
    
    # 1. RESET BUTTON
    if st.button("🗑️ Reset / Clear All", type="secondary"):
        clear_grid_state()
        st.rerun()

    st.markdown("---")
    
    # 2. GRID SIZE (With Callback)
    grid_size = st.radio(
        "Grid Size", 
        (4, 5), 
        index=0, 
        on_change=clear_grid_state
    )
    
    st.markdown("---")
    model_data = load_model()
    if model_data:
        st.success("✅ Brain Loaded")
    else:
        st.error("❌ Brain Missing")

# ==========================================
# 5. MAIN UI
# ==========================================
st.title("🧩 Gap Challenge Auto-Solver")

col1, col2 = st.columns([1, 1.2])

# --- LEFT COLUMN: INPUT ---
with col1:
    st.subheader("1. Input")
    
    # KEY TRICK: We attach grid_size to the key.
    # When grid_size changes, this becomes a "new" button, resetting the paste state entirely.
    paste_result = pbutton("📋 Paste Image", key=f"paste_btn_{grid_size}")
    
    if paste_result.image_data is not None:
        # Generate unique ID for this paste
        current_paste_id = hash(paste_result.image_data.tobytes())
        
        # New Paste Logic
        if current_paste_id != st.session_state.last_paste_id or st.session_state.grid_data is None:
            st.session_state.last_paste_id = current_paste_id
            
            with st.spinner(f"Solving {grid_size}x{grid_size}..."):
                # Convert to BGR for OpenCV
                img_array = np.array(paste_result.image_data.convert('RGB'))[:, :, ::-1]
                
                if model_data:
                    clf = model_data['model']
                    preds, cropped_bgr = predict_grid_labels(img_array, grid_size, clf)
                    
                    st.session_state.grid_data = preds
                    # Save RGB version for display (Fixes Color Shift)
                    st.session_state.cropped_view = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2RGB)
                else:
                    st.warning("Train model first.")
                    
        # Display the RGB view
        if 'cropped_view' in st.session_state:
            # Use 'use_container_width' to fix deprecation warning
            st.image(st.session_state.cropped_view, caption="Detected Board", use_container_width=True)
        else:
            st.image(paste_result.image_data, caption="Raw Paste", use_container_width=True)
            
    else:
        st.info(f"Paste a {grid_size}x{grid_size} screenshot to begin.")

# --- RIGHT COLUMN: EDITOR & SOLUTION ---
with col2:
    st.subheader("2. Verify & Solve")
    
    # Render only if valid data exists
    if st.session_state.grid_data and len(st.session_state.grid_data) == grid_size:
        
        shape_options = sorted(model_data['classes']) if model_data else ['1circle', '2triangle', '3square', '4cross', '5star']
        if 'blank' not in shape_options: shape_options.insert(0, 'blank')
        if '6question' not in shape_options: shape_options.append('6question')

        updated_grid = []
        
        # Grid Editor
        for r in range(grid_size):
            cols = st.columns(grid_size)
            row_data = []
            for c in range(grid_size):
                val = st.session_state.grid_data[r][c]
                # Include paste ID in key to force reset on new image
                new_val = cols[c].selectbox(
                    label=f"({r},{c})",
                    options=shape_options,
                    index=shape_options.index(val) if val in shape_options else 0,
                    key=f"c_{r}_{c}_{st.session_state.last_paste_id}",
                    label_visibility="collapsed"
                )
                row_data.append(new_val)
            updated_grid.append(row_data)
        
        st.session_state.grid_data = updated_grid
        
        # Auto-Solve
        st.markdown("---")
        result, final_board = run_solver_logic(updated_grid)
        
        if final_board:
            clean_name = str(result).replace("1","").replace("2","").replace("3","").replace("4","").replace("5","").replace("6","").upper()
            st.success(f"### Missing Shape: {clean_name}")
            with st.expander("View Full Solution Grid"):
                st.table(final_board)
        else:
            st.error(f"Status: {result}")
            st.warning("Check grid for errors.")
            
    elif st.session_state.grid_data:
        # Fallback if state gets messy
        st.warning("State mismatch. Click Reset.")

# ==========================================
# 6. HELP
# ==========================================
st.markdown("---")
with st.expander("ℹ️ How it works"):
    st.markdown("""
    **Gap Challenge Solver**
    1.  **Select Grid Size:** Toggle between 4x4 and 5x5. This will clear the current board.
    2.  **Paste Image:** Screenshot the puzzle. The app detects the board, fixes colors, and centers shapes automatically.
    3.  **Review:** If the AI mistakes a shape, correct it in the dropdowns.
    4.  **Solve:** The answer updates instantly below.
    """)