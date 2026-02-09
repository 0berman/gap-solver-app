import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import time
import glob

# ==========================================
# 0. SETUP FOLDERS
# ==========================================
BASE_DIR = "dataset"
LABELS = ['1circle', '2triangle', '3square', '4cross', '5star', '6question', 'blank']

# Create directories if they don't exist
os.makedirs(os.path.join(BASE_DIR, "unsorted"), exist_ok=True)
for l in LABELS:
    os.makedirs(os.path.join(BASE_DIR, l), exist_ok=True)

# ==========================================
# 1. SMART CROPPER (Macro: Finds the Board)
# ==========================================
def smart_crop_board(image, grid_n):
    """
    Detects the puzzle board within a screenshot using Adaptive Thresholding
    and clustering of cell-like contours.
    """
    img = image.copy()
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Adaptive Thresholding handles Dark Mode & Light Mode equally well
    thresh = cv2.adaptiveThreshold(
        gray, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 
        11, 2
    )

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Define expected cell size range to filter noise
    expected_cell_w = w // grid_n
    min_dim = expected_cell_w * 0.2
    max_dim = expected_cell_w * 1.5
    
    valid_boxes = []
    
    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        aspect = cw / float(ch)
        
        # Check if contour is roughly square and correct size
        if 0.7 < aspect < 1.3 and min_dim < cw < max_dim and min_dim < ch < max_dim:
            valid_boxes.append((x, y, cw, ch))
            
    # Fallback: If we can't find the grid, return original image
    if len(valid_boxes) < 4: 
        return img

    # Calculate the bounding box of the entire cluster of cells
    min_x = min([b[0] for b in valid_boxes])
    min_y = min([b[1] for b in valid_boxes])
    max_x = max([b[0] + b[2] for b in valid_boxes])
    max_y = max([b[1] + b[3] for b in valid_boxes])
    
    # Sanity check: prevent cropping to tiny noise clusters
    if (max_x - min_x) < w * 0.3 or (max_y - min_y) < h * 0.3:
        return img
        
    return img[min_y:max_y, min_x:max_x]

# ==========================================
# 2. THE SEEKER (Micro: Centers the Shape)
# ==========================================
def get_centered_roi(cell_image):
    """
    Looks inside a specific grid cell, finds the shape (blob), 
    and returns a tight, centered crop of it.
    """
    h, w = cell_image.shape[:2]
    
    # Quick threshold to find the shape blob
    gray = cv2.cvtColor(cell_image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_blobs = [c for c in contours if cv2.contourArea(c) > 50]
    
    # Define the output size we want (70% of the original cell size is usually good)
    target_w = int(w * 0.70)
    target_h = int(h * 0.70)
    
    if valid_blobs:
        # Find the center of all blobs combined (handles '?' which has 2 parts)
        min_bx = w
        min_by = h
        max_bx = 0
        max_by = 0
        
        for c in valid_blobs:
            bx, by, bw, bh = cv2.boundingRect(c)
            min_bx = min(min_bx, bx)
            min_by = min(min_by, by)
            max_bx = max(max_bx, bx + bw)
            max_by = max(max_by, by + bh)
            
        center_x = (min_bx + max_bx) // 2
        center_y = (min_by + max_by) // 2
        
        # Calculate crop coordinates centered on the shape
        x1 = max(0, center_x - target_w // 2)
        y1 = max(0, center_y - target_h // 2)
        x2 = min(w, x1 + target_w)
        y2 = min(h, y1 + target_h)
        
        # Boundary checks
        if x2 - x1 < target_w: x1 = max(0, x2 - target_w)
        if y2 - y1 < target_h: y1 = max(0, y2 - target_h)
        
        return cell_image[y1:y2, x1:x2]

    else:
        # If blank, just return the dead center
        margin_x = (w - target_w) // 2
        margin_y = (h - target_h) // 2
        return cell_image[margin_y:margin_y+target_h, margin_x:margin_x+target_w]

# ==========================================
# 3. HELPER: SHAPE GUESSER (For Sorting)
# ==========================================
def detect_shape_by_contour(cell_image):
    # Old logic used only for guessing folders
    roi = cell_image.copy()
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return "blank"

    valid_blobs = []
    roi_h, roi_w = roi.shape[:2]
    for c in contours:
        if cv2.contourArea(c) < 50: continue 
        _, _, w, h = cv2.boundingRect(c)
        if w > 0.95 * roi_w or h > 0.95 * roi_h: continue
        valid_blobs.append(c)

    if not valid_blobs: return "blank"
    if len(valid_blobs) >= 2: return "6question"

    largest_contour = max(valid_blobs, key=cv2.contourArea)
    area = cv2.contourArea(largest_contour)
    hull = cv2.convexHull(largest_contour)
    hull_area = cv2.contourArea(hull)
    solidity = float(area)/hull_area if hull_area > 0 else 0
    perimeter = cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, 0.035 * perimeter, True) 
    vertices = len(approx)
    is_convex = cv2.isContourConvex(approx)
    
    if is_convex or solidity > 0.92:
        _, _, w, h = cv2.boundingRect(largest_contour)
        extent = float(area) / (w * h) if w*h > 0 else 0
        if extent < 0.75:
            return "2triangle" if solidity > 0.9 else "4cross"
        if vertices == 4: return "3square"
        elif vertices == 3: return "2triangle"
        else: return "1circle"
    else:
        if vertices >= 11: return "4cross" if solidity > 0.55 else "6question"
        elif 9 <= vertices <= 10: return "5star"
        else:
            if solidity < 0.45: return "6question"
            elif solidity > 0.80: return "4cross"
            else: return "5star"

# ==========================================
# 4. MAIN PROCESSING PIPELINE
# ==========================================

def process_single_image(cv_image, grid_size, origin_name="img"):
    # STEP 1: Smart Crop (Find the board)
    board_img = smart_crop_board(cv_image, grid_size)
    
    img_h, img_w = board_img.shape[:2]
    cell_h = img_h // grid_size
    cell_w = img_w // grid_size
    
    saved_count = 0
    timestamp = int(time.time() * 1000)

    # STEP 2: Iterate Grid
    for r in range(grid_size):
        for c in range(grid_size):
            y1 = r * cell_h
            x1 = c * cell_w
            
            # Extract Raw Cell
            raw_cell = board_img[y1:y1+cell_h, x1:x1+cell_w]
            
            # STEP 3: The Seeker (Micro-Crop)
            # Find the shape and re-center the camera on it
            centered_cell = get_centered_roi(raw_cell)
            
            # Auto-Sort Guess (Run on the clean centered image)
            try:
                guessed_label = detect_shape_by_contour(centered_cell) 
            except:
                guessed_label = "unsorted"
            
            if guessed_label not in LABELS: guessed_label = "unsorted"
            
            # Save
            safe_origin = origin_name.split('.')[0][-10:]
            filename = f"{guessed_label}_{timestamp}_{safe_origin}_{r}_{c}.png"
            save_path = os.path.join(BASE_DIR, guessed_label, filename)
            
            cv2.imwrite(save_path, centered_cell)
            saved_count += 1
            
    return saved_count

# ==========================================
# 5. STREAMLIT UI
# ==========================================

st.set_page_config(page_title="Batch Processor Pro", layout="wide", page_icon="🏭")
st.title("🏭 Batch Data Processor: Final Edition")
st.markdown("""
**Robustness Features Active:**
* ✅ **Smart Board Detection:** Ignores browser margins & toolbars.
* ✅ **Dark Mode Support:** Uses adaptive thresholding.
* ✅ **The Seeker:** Automatically finds shapes and re-centers the crop.
""")

with st.sidebar:
    st.header("Settings")
    grid_size = st.radio("Grid Size for Batch", (4, 5), index=0)

tab1, tab2 = st.tabs(["📂 Process Folder", "⬆️ Upload Files"])

# --- TAB 1: LOCAL FOLDER ---
with tab1:
    st.write("Point to a folder on your computer containing screenshots.")
    default_path = os.path.join(os.getcwd(), "raw_images")
    folder_path = st.text_input("Folder Path:", value=default_path)
    
    if st.button("Start Folder Batch", type="primary"):
        if not os.path.exists(folder_path):
            st.error("Folder not found!")
        else:
            extensions = ['*.png', '*.jpg', '*.jpeg']
            files = []
            for ext in extensions:
                files.extend(glob.glob(os.path.join(folder_path, ext)))
            
            if not files:
                st.warning("No images found.")
            else:
                progress_bar = st.progress(0)
                status_text = st.empty()
                total_cells = 0
                
                for i, file_path in enumerate(files):
                    img = cv2.imread(file_path)
                    if img is not None:
                        f_name = os.path.basename(file_path)
                        count = process_single_image(img, grid_size, f_name)
                        total_cells += count
                    
                    progress_bar.progress((i + 1) / len(files))
                    status_text.text(f"Processing: {os.path.basename(file_path)}")
                
                st.success(f"🎉 Processed {len(files)} images -> Extracted {total_cells} cells!")
                st.balloons()

# --- TAB 2: UPLOAD FILES ---
with tab2:
    st.write("Drag and drop multiple files here.")
    uploaded_files = st.file_uploader("Upload Screenshots", accept_multiple_files=True, type=['png', 'jpg'])
    
    if uploaded_files and st.button("Process Uploaded Files"):
        progress_bar = st.progress(0)
        total_cells = 0
        
        for i, uploaded_file in enumerate(uploaded_files):
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, 1)
            
            count = process_single_image(img, grid_size, uploaded_file.name)
            total_cells += count
            
            progress_bar.progress((i + 1) / len(uploaded_files))
            
        st.success(f"Done! Saved {total_cells} cells.")

# --- STATS ---
st.markdown("---")
st.subheader("Dataset Status")
cols = st.columns(len(LABELS))
for i, l in enumerate(LABELS):
    path = os.path.join(BASE_DIR, l)
    count = len(os.listdir(path))
    cols[i].metric(l, count)