import cv2
import numpy as np
import joblib
import os
import glob
import time

# ==========================================
# CONFIGURATION
# ==========================================
# Folder containing your raw screenshots (e.g. "raw_screenshots")
INPUT_FOLDER = r"C:\Github Code\gap-solver-app\examples\new_5"

# Where to save the sorted crops
OUTPUT_BASE = r"C:\Github Code\gap-solver-app\new_datasets"

# Grid size to assume (usually 5 for the hard levels)
GRID_SIZE = 5

# Load your current brain
if not os.path.exists("C:\Github Code\gap-solver-app\phase3-SVM\model.pkl"):
    print("❌ Error: model.pkl not found!")
    exit()

model_data = joblib.load("C:\Github Code\gap-solver-app\phase3-SVM\model.pkl")
clf = model_data['model']
classes = model_data['classes']

# Import the exact logic the app uses
# (Make sure preprocess.py is in the same folder)
from preprocess import standardize_cell

# ==========================================
# HELPER: MACRO CROPPER
# ==========================================
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

# ==========================================
# MAIN LOOP
# ==========================================
def main():
    # 1. Create Output Folders
    for label in classes:
        os.makedirs(os.path.join(OUTPUT_BASE, label), exist_ok=True)
    
    # 2. Get Images
    types = ('*.png', '*.jpg', '*.jpeg')
    files = []
    for ext in types:
        files.extend(glob.glob(os.path.join(INPUT_FOLDER, ext)))
    
    if not files:
        print(f"⚠️ No images found in '{INPUT_FOLDER}'. Please create this folder and put screenshots in it.")
        return

    print(f"🚀 Processing {len(files)} screenshots...")
    total_cells = 0

    for file_path in files:
        filename = os.path.basename(file_path).split('.')[0]
        img = cv2.imread(file_path)
        if img is None: continue
        
        # A. Find Board
        board = smart_crop_board(img, GRID_SIZE)
        h, w = board.shape[:2]
        cell_h, cell_w = h // GRID_SIZE, w // GRID_SIZE
        
        # B. Iterate Cells
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                y1, x1 = r*cell_h, c*cell_w
                
                # C. Slice (Exact logic from App)
                m_h = int(cell_h * 0.05)
                m_w = int(cell_w * 0.05)
                raw_cell = board[y1:y1+cell_h, x1:x1+cell_w]
                safe_cell = raw_cell[m_h:cell_h-m_h, m_w:cell_w-m_w]
                
                # D. Preprocess & Predict
                # We need to standardize to get a prediction...
                try:
                    clean_input = standardize_cell(safe_cell)
                    feat = clean_input.flatten() / 255.0
                    pred_label = clf.predict([feat])[0]
                except:
                    pred_label = "unsorted"

                # E. SAVE THE RAW IMAGE (Not the clean one!)
                # We save 'safe_cell' because we want to train the AI 
                # to handle the shifting/messiness of this specific crop.
                timestamp = int(time.time() * 1000)
                save_name = f"{filename}_r{r}c{c}_{timestamp}.png"
                save_path = os.path.join(OUTPUT_BASE, pred_label, save_name)
                
                cv2.imwrite(save_path, safe_cell)
                total_cells += 1
                
        print(f"✅ Processed: {filename}")

    print(f"\n🎉 Done! Extracted {total_cells} cells into '{OUTPUT_BASE}'.")
    print("👉 ACTION REQUIRED: Go into that folder, verify the images are in the correct folders, then move them to your main 'dataset'.")

if __name__ == "__main__":
    main()