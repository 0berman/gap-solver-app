import cv2
import numpy as np
import os

# ==========================================
# CONFIGURATION
# ==========================================
IMAGE_PATH = r"C:\Github Code\gap-solver-app\templates\Examples\row_Example6.png" # Update this for testing
GRID_SIZE = 4  # Toggle between 4 and 5

# ==========================================
# 1. SMART CROPPER (Dilation + Padding)
# ==========================================
def crop_to_grid(source_image):
    """
    1. Uses DILATION to merge separate shapes into a single 'grid block'.
    2. Finds the bounding box of that block.
    3. Adds PADDING to preserve cell margins.
    4. Force-Crops bottom to ensure square aspect ratio.
    """
    gray = cv2.cvtColor(source_image, cv2.COLOR_BGR2GRAY)
    
    # Invert so content is white, bg is black (Standard method, robust to colors)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # --- FIX 1: DILATION (The "Fat Blob" Trick) ---
    # We dilate the shapes heavily so they connect or define the "Grid Area"
    # rather than just the "Tip of the Star".
    kernel = np.ones((10,10), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=2)
    
    non_zero_pixels = cv2.findNonZero(dilated)
    if non_zero_pixels is None: return source_image
        
    x, y, w, h = cv2.boundingRect(non_zero_pixels)
    
    # --- FIX 2: SAFETY PADDING ---
    # We assume the detected box is "tight" to the shapes. 
    # We add ~2% padding outwards to capture the cell boundary/whitespace.
    pad = int(w * 0.03) 
    
    # Apply padding with bounds checking
    img_h, img_w = source_image.shape[:2]
    x = max(0, x - pad)
    y = max(0, y - pad)
    w = min(img_w - x, w + (pad * 2))
    
    # --- FIX 3: FORCE SQUARE (Bottom Crop) ---
    # We trust the Width (w) more than the Height (h) because of the bottom bar.
    # We set the crop height equal to the width.
    crop_h = w
    
    # Clip if it goes off bottom
    if y + crop_h > img_h:
        crop_h = img_h - y
        
    print(f"DEBUG: Grid detected at {x},{y}. Cropping to {w}x{crop_h} (Square Forced)")
    return source_image[y:y+crop_h, x:x+w]

# ==========================================
# 2. SHAPE RECOGNITION (Tuned for V:8 Question Mark)
# ==========================================
def detect_shape_by_contour(cell_image):
    # PRE-PROCESSING
    gray = cv2.cvtColor(cell_image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # OTSU THRESHOLDING (Best for solid shapes)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # FRAME REJECTION (Peel off outer square frames)
    h, w = thresh.shape
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    valid_blobs = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < 200: continue 
        
        # Check if this blob is a Frame (touches edges or is huge)
        bx, by, bw, bh = cv2.boundingRect(c)
        if bw > 0.85 * w or bh > 0.85 * h:
            continue
        valid_blobs.append(c)
        
    if not valid_blobs:
        return "blank", "Frame/Noise Only", thresh

    # GEOMETRY ANALYSIS
    largest_contour = max(valid_blobs, key=cv2.contourArea)
    area = cv2.contourArea(largest_contour)
    
    mask = np.zeros_like(thresh)
    cv2.drawContours(mask, [largest_contour], -1, 255, -1)
    
    perimeter = cv2.arcLength(largest_contour, True)
    epsilon = 0.03 * perimeter 
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)
    vertices = len(approx)
    
    hull = cv2.convexHull(largest_contour)
    hull_area = cv2.contourArea(hull)
    solidity = float(area)/hull_area if hull_area > 0 else 0
    
    circularity = 0
    if perimeter > 0:
        circularity = (4 * np.pi * area) / (perimeter ** 2)
        
    is_convex = cv2.isContourConvex(approx)

    # --- DECISION TREE ---
    label = "6question" # Default

    # BRANCH A: SIMPLE CONVEX SHAPES
    if is_convex:
        if vertices == 3:
            label = "2triangle"
        elif vertices == 4:
            x,y,bw,bh = cv2.boundingRect(approx)
            aspect = float(bw)/bh
            label = "3square" if 0.8 < aspect < 1.2 else "3square"
        else:
            label = "1circle"
            
    # BRANCH B: COMPLEX SHAPES (Non-Convex)
    else:
        # Override: Messy Squares sometimes look non-convex but are very solid
        if solidity > 0.95 and vertices <= 6:
             label = "3square"
             
        elif vertices >= 12:
            label = "4cross"
            
        elif 9 <= vertices <= 11:
            label = "5star"
            
        elif vertices == 8:
             # --- FIX: RELAXED QUESTION MARK RULE ---
             # Your data: S=0.69, C=0.35. 
             # Previous rule (C < 0.25) failed. New rule (C < 0.40) catches it.
             if circularity < 0.40: 
                 label = "6question"
             elif solidity > 0.7: 
                 label = "4cross"
             else: 
                 label = "5star"
             
        elif 5 <= vertices <= 7:
            # Danger Zone: Star vs Question Mark Hook
            label = "5star" if circularity > 0.45 else "6question"

    # BRANCH C: MULTI-PART OVERRIDE
    if label not in ["1circle", "3square", "2triangle"]:
        if len(valid_blobs) >= 2:
            sorted_blobs = sorted(valid_blobs, key=cv2.contourArea, reverse=True)
            if cv2.contourArea(sorted_blobs[1]) > 50:
                label = "6question"

    debug_info = f"V:{vertices} S:{solidity:.2f} C:{circularity:.2f} B:{len(valid_blobs)}"
    return label, debug_info, mask

# ==========================================
# MAIN LOOP
# ==========================================
def main():
    if not os.path.exists(IMAGE_PATH):
        print(f"❌ Error: Image not found at {IMAGE_PATH}")
        return

    original_img = cv2.imread(IMAGE_PATH)
    
    # 1. CROP (Using new Robust Logic)
    # Add large border first to ensure dilation doesn't hit image edge
    border_size = 50 
    bordered_img = cv2.copyMakeBorder(original_img, border_size, border_size, border_size, border_size, cv2.BORDER_CONSTANT, value=[255,255,255])
    cropped_img = crop_to_grid(bordered_img)
    
    # 2. RESIZE
    standard_size = 600
    display_img = cv2.resize(cropped_img, (standard_size, standard_size))
    
    cell_h = standard_size // GRID_SIZE
    cell_w = standard_size // GRID_SIZE

    print("\n--- VISUAL DEBUGGER (ROBUST CROP MODE) ---")
    print(f"Grid: {GRID_SIZE}x{GRID_SIZE}")
    print("Press SPACE to next cell. 'q' to quit.")

    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            y1, y2 = r * cell_h, (r + 1) * cell_h
            x1, x2 = c * cell_w, (c + 1) * cell_w
            
            # Margin (Small margin to avoid shared edges)
            m = 8 
            cell_roi = display_img[y1+m:y2-m, x1+m:x2-m]
            
            label, stats, debug_thresh = detect_shape_by_contour(cell_roi)
            
            # VISUALIZATION
            view = display_img.copy()
            cv2.rectangle(view, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(view, f"Pred: {label}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(view, stats, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            roi_zoom = cv2.resize(cell_roi, (300, 300))
            thresh_zoom = cv2.resize(debug_thresh, (300, 300))
            thresh_zoom_bgr = cv2.cvtColor(thresh_zoom, cv2.COLOR_GRAY2BGR)
            
            cv2.imshow("1. Grid Alignment", view)
            cv2.imshow("2. Cell Analysis", np.hstack((roi_zoom, thresh_zoom_bgr)))
            
            if cv2.waitKey(0) == ord('q'):
                cv2.destroyAllWindows()
                return

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()