import cv2
import numpy as np
import os

# ==========================================
# CONFIGURATION
# ==========================================
IMAGE_PATH = r"C:\Github Code\gap-solver-app\templates\Examples\Example5.png"
GRID_SIZE = 4

# ==========================================
# 1. ROBUST CROPPER (Edge-Based Alignment)
# ==========================================
def crop_to_grid(source_image, grid_n=GRID_SIZE):
    """
    Robust Grid Cropper:
    1. Detects edges to find the gray square 'cells'.
    2. Filters for valid cell-like squares.
    3. Sorts them top-to-bottom.
    4. Picks the top N*N cells (ignoring the bottom row).
    5. Crops to the exact bounding box of those cells.
    """
    # Work on a copy
    img = source_image.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. Edge Detection (Better than thresholding for light gray boxes)
    #    The gray boxes have a distinct edge against the white background.
    edges = cv2.Canny(gray, 50, 150)
    
    # 2. Find Contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    valid_cells = []
    img_h, img_w = img.shape[:2]
    min_area = (img_w // 20) ** 2  # Dynamic min size (avoid noise)
    
    for c in contours:
        # Approximate geometric shape
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        x, y, w, h = cv2.boundingRect(approx)
        area = w * h
        aspect_ratio = w / float(h)
        
        # FILTER: Look for the Grid Cells
        # - Must have 4 corners (square-ish)
        # - Must be a reasonable size
        # - Must be roughly square (aspect ratio ~1.0)
        if len(approx) == 4 and area > min_area and 0.8 < aspect_ratio < 1.2:
            valid_cells.append((x, y, w, h))

    # 3. Sort and Select
    if not valid_cells:
        print("❌ Warning: No grid cells found. Returning original.")
        return source_image

    # Sort primarily by Y (top to bottom), secondarily by X (left to right)
    # This groups the top grid separately from the bottom row.
    valid_cells.sort(key=lambda k: (k[1], k[0]))
    
    # We expect N*N cells for the main grid. 
    # Even if we detect the bottom row, we only take the top N*N.
    expected_cells = grid_n * grid_n
    main_grid_cells = valid_cells[:expected_cells]
    
    # 4. Calculate Crop Bounds based on the selected cells
    # Find the extreme edges of just the main grid cells
    min_x = min([c[0] for c in main_grid_cells])
    min_y = min([c[1] for c in main_grid_cells])
    max_x = max([c[0] + c[2] for c in main_grid_cells])
    max_y = max([c[1] + c[3] for c in main_grid_cells])
    
    # Add a tiny bit of padding (optional, e.g., 5 pixels)
    pad = 5
    min_x = max(0, min_x - pad)
    min_y = max(0, min_y - pad)
    max_x = min(img_w, max_x + pad)
    max_y = min(img_h, max_y + pad)

    print(f"DEBUG: Found {len(valid_cells)} cells. Keeping top {len(main_grid_cells)}. Crop: {min_x},{min_y} to {max_x},{max_y}")
    
    return source_image[min_y:max_y, min_x:max_x]

# ==========================================
# 2. SHAPE RECOGNITION (Hierarchy & Peeling)
# ==========================================
def detect_shape_by_contour(cell_image):
    # --- 1. INNER CROP (Relaxed to 10%) ---
    h_full, w_full = cell_image.shape[:2]
    # Reduced margin to 10% to prevent cutting off thick cross arms
    margin_x = int(w_full * 0.15)
    margin_y = int(h_full * 0.15)
    
    roi = cell_image[margin_y:h_full-margin_y, margin_x:w_full-margin_x]
    
    # --- Standard Pre-processing ---
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return "blank", "Empty", thresh

    valid_blobs = []
    roi_h, roi_w = roi.shape[:2]
    
    for c in contours:
        area = cv2.contourArea(c)
        if area < 50: continue 
        
        x, y, w, h = cv2.boundingRect(c)
        if w > 0.95 * roi_w or h > 0.95 * roi_h: continue
        valid_blobs.append(c)

    if not valid_blobs:
        return "blank", "Noise/Empty", thresh

    # --- 2. PRIORITY CHECK: BLOB COUNT ---
    if len(valid_blobs) >= 2:
        return "6question", f"Blobs: {len(valid_blobs)}", thresh

    # --- 3. GEOMETRY ANALYSIS ---
    largest_contour = max(valid_blobs, key=cv2.contourArea)
    area = cv2.contourArea(largest_contour)
    
    hull = cv2.convexHull(largest_contour)
    hull_area = cv2.contourArea(hull)
    solidity = float(area)/hull_area if hull_area > 0 else 0
    
    x, y, w, h = cv2.boundingRect(largest_contour)
    rect_area = w * h
    extent = float(area) / rect_area if rect_area > 0 else 0
    
    perimeter = cv2.arcLength(largest_contour, True)
    # Tweak: Slightly lower epsilon (0.03) to try and catch the cross corners better
    approx = cv2.approxPolyDP(largest_contour, 0.035 * perimeter, True) 
    vertices = len(approx)
    is_convex = cv2.isContourConvex(approx)

    debug_info = f"V:{vertices} Ext:{extent:.2f} Sol:{solidity:.2f}"

    # --- 4. DECISION TREE (Final Robust Version) ---
    
    # Logic Group A: The "Blocky" Shapes
    # If it is Convex OR has very high solidity (like a circle)
    if is_convex or solidity > 0.92:
        
        # LOW EXTENT CHECK (The Fix is Here)
        if extent < 0.75:
            # Both Triangles and "Diamond-like" Crosses land here.
            # Differentiator: Solidity.
            # Triangle is solid (~0.95+). Cross has armpits (~0.8).
            if solidity > 0.9:
                return "2triangle", debug_info, thresh
            else:
                return "4cross", debug_info, thresh
        
        # High Extent Logic
        if vertices == 4: return "3square", debug_info, thresh
        elif vertices == 3: return "2triangle", debug_info, thresh
        else: return "1circle", debug_info, thresh

    # Logic Group B: The "Complex" Shapes
    else:
        # Cross vs Question Mark
        if vertices >= 11:
            if solidity > 0.55: return "4cross", debug_info, thresh
            else: return "6question", debug_info, thresh
            
        elif 9 <= vertices <= 10:
            return "5star", debug_info, thresh
            
        # Ambiguous Vertex Counts (smoothed shapes)
        else:
            if solidity < 0.45: return "6question", debug_info, thresh
            # If solidity is medium-high (0.6 - 0.8), it's likely a Cross or Star
            # Star usually has lower solidity than cross due to sharp points
            elif solidity > 0.80: return "4cross", debug_info, thresh
            else: return "5star", debug_info, thresh

# ==========================================
# MAIN LOOP (Debugger)
# ==========================================
def main():
    if not os.path.exists(IMAGE_PATH): return

    original_img = cv2.imread(IMAGE_PATH)
    
    # Pre-add border to ensure grid edges are detectable
    bordered = cv2.copyMakeBorder(original_img, 40, 40, 40, 40, cv2.BORDER_CONSTANT, value=[255,255,255])
    cropped_img = crop_to_grid(bordered)
    
    standard_size = 600
    display_img = cv2.resize(cropped_img, (standard_size, standard_size))
    
    cell_h = standard_size // GRID_SIZE
    cell_w = standard_size // GRID_SIZE

    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            y1, y2 = r * cell_h, (r + 1) * cell_h
            x1, x2 = c * cell_w, (c + 1) * cell_w
            
            m = 6 # Small margin to clear the grid lines themselves
            cell_roi = display_img[y1+m:y2-m, x1+m:x2-m]
            
            label, stats, debug_mask = detect_shape_by_contour(cell_roi)
            
            view = display_img.copy()
            cv2.rectangle(view, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(view, f"Pred: {label}", (10, 30), 1, 1.5, (0,0,255), 2)
            cv2.putText(view, stats, (10, 60), 1, 1.2, (255,0,0), 2)

            roi_zoom = cv2.resize(cell_roi, (300, 300))
            mask_zoom = cv2.resize(debug_mask, (300, 300))
            mask_zoom_bgr = cv2.cvtColor(mask_zoom, cv2.COLOR_GRAY2BGR)
            
            cv2.imshow("Debugger", np.hstack((roi_zoom, mask_zoom_bgr)))
            cv2.imshow("Full Grid Alignment", view)
            
            if cv2.waitKey(0) == ord('q'): break
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()