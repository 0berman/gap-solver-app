import cv2
import numpy as np
import joblib
import os

# ==========================================
# CONFIGURATION
# ==========================================
# CHANGE THIS TO YOUR SCREENSHOT PATH
IMAGE_PATH = r"C:\Github Code\gap-solver-app\examples\xx) done\5\Screenshot 2026-02-04 101304.png"
GRID_SIZE = 5        
# Load Model
MODEL_PATH = r"C:\Github Code\gap-solver-app\phase3-SVM\model.pkl"
if not os.path.exists(MODEL_PATH):
    print("❌ Error: model.pkl not found.")
    exit()

model_data = joblib.load(MODEL_PATH)
clf = model_data['model']
classes = model_data['classes']

# ==========================================
# 1. PREPROCESSING (The "Universal Translator")
# ==========================================
def standardize_cell(img):
    # Resize to 64x64
    img = cv2.resize(img, (64, 64))
    
    # Blur
    blurred = cv2.GaussianBlur(img, (3, 3), 0)
    
    # Corner Difference Strategy
    corner_TL = blurred[0:3, 0:3]
    corner_TR = blurred[0:3, -3:]
    corner_BL = blurred[-3:, 0:3]
    corner_BR = blurred[-3:, -3:]
    corners = np.vstack((corner_TL, corner_TR, corner_BL, corner_BR))
    avg_bg_color = np.mean(corners, axis=(0, 1))
    
    # Difference Calculation
    diff = cv2.absdiff(blurred, avg_bg_color.astype(np.uint8))
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    
    # Threshold
    _, mask = cv2.threshold(diff_gray, 30, 255, cv2.THRESH_BINARY)
    
    # Cleanup
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2,2), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8))
    
    # Safety Border
    h, w = mask.shape
    cv2.rectangle(mask, (0,0), (w, h), 0, 2)
    
    # Re-Center
    coords = cv2.findNonZero(mask)
    final_img = np.zeros((64, 64), dtype=np.uint8)
    
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        shape = mask[y:y+h, x:x+w]
        center_x = (64 - w) // 2
        center_y = (64 - h) // 2
        final_img[center_y:center_y+h, center_x:center_x+w] = shape
        
    return final_img

# ==========================================
# 2. YOUR ORIGINAL SMART CROPPER
# ==========================================
def smart_crop_board(image, grid_n):
    img = image.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Adaptive Thresholding (Your preferred logic)
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 11, 2
    )
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    expected_w = img.shape[1] // grid_n
    valid_boxes = []
    
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        # Your specific aspect ratio and width filters
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
# 3. DIAGNOSTIC DASHBOARD
# ==========================================
def main():
    if not os.path.exists(IMAGE_PATH):
        print("Image file not found!")
        return

    original_img = cv2.imread(IMAGE_PATH)
    
    # 1. Find Board using YOUR logic
    board = smart_crop_board(original_img, GRID_SIZE)
    h, w = board.shape[:2]
    cell_h, cell_w = h // GRID_SIZE, w // GRID_SIZE
    
    print(f"Board Found: {w}x{h}. Press SPACE for next cell.")

    # Create Dashboard Canvas
    dash_h, dash_w = 700, 1200
    dashboard = np.zeros((dash_h, dash_w, 3), dtype=np.uint8)

    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            y1, x1 = r*cell_h, c*cell_w
            y2, x2 = y1+cell_h, x1+cell_w
            
            # Boundary Check
            if y2 > h or x2 > w: continue

            # Extract (with your 5% margin logic)
            m_h = int(cell_h * 0.05)
            m_w = int(cell_w * 0.05)
            raw_cell = board[y1:y1+cell_h, x1:x1+cell_w]
            safe_cell = raw_cell[m_h:cell_h-m_h, m_w:cell_w-m_w]
            
            # Process & Predict
            svm_input = standardize_cell(safe_cell)
            feat = svm_input.flatten() / 255.0
            pred_idx = clf.predict([feat])[0]
            label = str(pred_idx)
            
            # --- BUILD VISUALIZATION ---
            dashboard[:] = (30, 30, 30) # Dark Gray Background
            
            # 1. LEFT PANEL: Full Board Context
            scale = 600 / h
            disp_w = int(w * scale)
            board_disp = cv2.resize(board, (disp_w, 600))
            
            # Draw Green Box around current cell
            bx1 = int(x1 * scale)
            by1 = int(y1 * scale)
            bx2 = int(x2 * scale)
            by2 = int(y2 * scale)
            cv2.rectangle(board_disp, (bx1, by1), (bx2, by2), (0, 255, 0), 4)
            
            # Paste Board into Dashboard
            dashboard[50:650, 50:50+disp_w] = board_disp
            cv2.putText(dashboard, "1. GLOBAL CONTEXT", (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

            # 2. TOP RIGHT: Raw Cell Zoom
            zoom_size = 280
            raw_zoom = cv2.resize(safe_cell, (zoom_size, zoom_size))
            x_right = 50 + disp_w + 50
            dashboard[50:50+zoom_size, x_right:x_right+zoom_size] = raw_zoom
            cv2.rectangle(dashboard, (x_right, 50), (x_right+zoom_size, 50+zoom_size), (255, 255, 255), 2)
            cv2.putText(dashboard, "2. HUMAN VIEW (Raw)", (x_right, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            # 3. BOTTOM RIGHT: SVM Input (The "Brain")
            svm_zoom = cv2.resize(svm_input, (zoom_size, zoom_size), interpolation=cv2.INTER_NEAREST)
            svm_zoom_color = cv2.cvtColor(svm_zoom, cv2.COLOR_GRAY2BGR)
            
            y_svm = 50 + zoom_size + 40
            dashboard[y_svm:y_svm+zoom_size, x_right:x_right+zoom_size] = svm_zoom_color
            cv2.rectangle(dashboard, (x_right, y_svm), (x_right+zoom_size, y_svm+zoom_size), (0, 255, 255), 2)
            cv2.putText(dashboard, "3. MACHINE VIEW (Preprocessed)", (x_right, y_svm-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

            # 4. PREDICTION LABEL
            cv2.putText(dashboard, f"PREDICTION: {label}", (x_right, y_svm + zoom_size + 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

            cv2.imshow("Act 3: SVM Diagnostic Center", dashboard)
            
            key = cv2.waitKey(0)
            if key == 27: # ESC
                cv2.destroyAllWindows()
                return

    cv2.destroyAllWindows()
    print("Debugging Complete.")

if __name__ == "__main__":
    main()