import cv2
import numpy as np
import os

# ==========================================
# CONFIGURATION
# ==========================================
# UPDATE THIS PATH TO YOUR TEST IMAGE
IMAGE_PATH = r"C:\Github Code\gap-solver-app\examples\5\Screenshot 2026-02-04 142307.png"
GRID_SIZE = 5

# ==========================================
# 1. CANNY CROPPER (Adaptive Threshold Version)
# ==========================================
def crop_to_grid(source_image, grid_n):
    img = source_image.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. Preprocessing
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Adaptive Thresholding 
    thresh = cv2.adaptiveThreshold(blurred, 255, 
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)
    
    # 2. Find Contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    valid_cells = []
    img_h, img_w = img.shape[:2]
    
    # Initial Filter
    min_area_threshold = (img_w // 50) ** 2 
    
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        x, y, w, h = cv2.boundingRect(approx)
        area = w * h
        aspect_ratio = w / float(h)
        
        # Relaxed Constraints
        if area > min_area_threshold and 0.8 < aspect_ratio < 1.2:
            valid_cells.append((x, y, w, h))

    if not valid_cells:
        print("Detection failed. Returning original.")
        return source_image

    # 3. Intelligent Filtering (Stricter)
    areas = [c[2] * c[3] for c in valid_cells]
    median_area = np.median(areas)
    
    filtered_cells = []
    for (x,y,w,h) in valid_cells:
        area = w*h
        if 0.7 * median_area < area < 1.3 * median_area:
            filtered_cells.append((x,y,w,h))
            
    if not filtered_cells:
        return source_image

    # 4. Calculate Bounds
    min_x = min([c[0] for c in filtered_cells])
    min_y = min([c[1] for c in filtered_cells])
    max_x = max([c[0] + c[2] for c in filtered_cells])
    max_y = max([c[1] + c[3] for c in filtered_cells])
    
    # Add padding
    pad = 5
    min_x = max(0, min_x - pad)
    min_y = max(0, min_y - pad)
    max_x = min(img_w, max_x + pad)
    max_y = min(img_h, max_y + pad)
    
    if (max_x - min_x) < 50 or (max_y - min_y) < 50:
        return source_image
    
    return source_image[min_y:max_y, min_x:max_x]

# ==========================================
# 2. METRICS & LOGIC (UPDATED)
# ==========================================
def calculate_metrics(contour, crop_w, crop_h):
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = float(area) / hull_area if hull_area > 0 else 0
    
    # --- FIX 1: LOWER EPSILON ---
    # Was 0.035. Reduced to 0.02 to detect the "valleys" of the star/cross
    epsilon = 0.02 * perimeter 
    approx = cv2.approxPolyDP(contour, epsilon, True)
    vertices = len(approx)
    
    is_convex = cv2.isContourConvex(approx)
    
    x, y, w, h = cv2.boundingRect(contour)
    rect_area = w * h
    extent = float(area) / rect_area if rect_area > 0 else 0
    
    circularity = 0
    if perimeter > 0:
        circularity = (4 * np.pi * area) / (perimeter ** 2)

    aspect_ratio = float(w) / h if h > 0 else 0
    width_ratio = w / float(crop_w)
    
    return {
        "Vertices": vertices, "Solidity": solidity, "Extent": extent,
        "IsConvex": is_convex, "Area": area, "Circularity": circularity,
        "AspectRatio": aspect_ratio, "WidthRatio": width_ratio
    }

def get_decision(metrics, num_blobs):
    v = metrics["Vertices"]
    sol = metrics["Solidity"]
    conv = metrics["IsConvex"]
    ext = metrics["Extent"]
    circ = metrics["Circularity"]
    wr = metrics["WidthRatio"]
    
    # 1. BORDER & NOISE GUARD
    if wr > 0.85: return "Empty", f"Touches Edges ({wr:.2f})"
    if wr < 0.20: return "Empty", f"Too Small ({wr:.2f})" 
    
    # 2. BLOB CHECK
    if num_blobs >= 2: return "Question", f"Blob Count: {num_blobs}"

    # 3. VERTEX SANITY CHECK
    if v < 3: return "Noise", f"Not enough vertices ({v})"

    # 4. CIRCLE CHECK
    # Note: With lower epsilon, circles might have 8-12 vertices, 
    # but they are still convex and have high circularity.
    if (conv or sol > 0.95) and circ > 0.82: # Lowered slightly to 0.82 to be safe
        if v == 4: return "Square", "V=4 (High Circ)"
        return "Circle", f"High Circ: {circ:.2f}"

    # 5. DECISION TREE
    if conv or sol > 0.92:
        # --- FIX 2: HANDLE PENTAGON STAR ---
        if v == 5: return "Star", "5 Vertices (Pentagon)" 

        if ext < 0.75:
            if sol > 0.9: return "Triangle", "Solid + Low Extent"
            else: return "Cross", "Solid + Low Extent (Non-Tri)"
        
        if v == 4: return "Square", "Solid + 4 Vertices"
        elif v == 3: return "Triangle", "Solid + 3 Vertices"
        else:
            if circ > 0.8: return "Circle", "Fallback Circle"
            return "Triangle", "Fallback Triangle"
            
    else:
        # Non-Convex Shapes (Stars, Crosses)
        # With epsilon 0.02, Stars will have ~10 vertices, Crosses ~12
        if v >= 11:
            if sol > 0.65: return "Cross", "V>=11 + Solid" # Cross is blockier
            else: return "Star", "V>=11 + Spiky" # Star is spikier
        
        elif 9 <= v <= 10:
            return "Star", "9-10 Vertices"
            
        else:
            # Fallback logic
            if sol < 0.45: return "Question", "Low Solidity"
            elif sol > 0.75: return "Cross", "High Solidity"
            else: return "Star", "Mid Solidity"

    return "Unknown", "No Match"

# ==========================================
# 3. LIVE SCANNER (Widescreen UI)
# ==========================================
def run_live_scanner(image_path):
    if not os.path.exists(image_path):
        print(f"Error: {image_path} not found")
        return

    original = cv2.imread(image_path)
    board_img = crop_to_grid(original, GRID_SIZE)
    h, w = board_img.shape[:2]
    cell_h, cell_w = h // GRID_SIZE, w // GRID_SIZE
    
    print("Controls: [SPACE] Next Cell | [ESC] Quit")
    
    # Increased Dashboard Size
    dashboard_h, dashboard_w = 900, 1600
    dashboard = np.zeros((dashboard_h, dashboard_w, 3), dtype=np.uint8)

    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            y1, x1 = r * cell_h, c * cell_w
            y2, x2 = y1 + cell_h, x1 + cell_w
            
            if y2 > h or x2 > w: continue
                
            cell = board_img[y1:y2, x1:x2]
            margin_h = int(cell_h * 0.15)
            margin_w = int(cell_w * 0.15)
            safe_cell = cell[margin_h:cell_h-margin_h, margin_w:cell_w-margin_w]
            safe_h, safe_w = safe_cell.shape[:2]
            
            gray = cv2.cvtColor(safe_cell, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            valid_blobs = [cnt for cnt in contours if cv2.contourArea(cnt) > 15]
            num_blobs = len(valid_blobs)
            
            best_cnt = None
            if valid_blobs:
                best_cnt = max(valid_blobs, key=cv2.contourArea)
            
            zoom_view = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
            label, reason = "Empty", "No Content"
            metrics = {k: 0 for k in ["Vertices", "Solidity", "Extent", "IsConvex", "Circularity", "WidthRatio"]}
            
            if best_cnt is not None:
                metrics = calculate_metrics(best_cnt, safe_w, safe_h)
                label, reason = get_decision(metrics, num_blobs)
                
                cv2.drawContours(zoom_view, [best_cnt], -1, (0, 255, 0), 2)
                for blob in valid_blobs:
                    if blob is not best_cnt:
                        cv2.drawContours(zoom_view, [blob], -1, (0, 255, 255), -1)

            # --- BUILD UI (Compact Version) ---
            scale = 600 / h  # Scale board to 600px height
            disp_w = int(w * scale)
            sidebar_w = 350  # Fixed width for stats
            padding = 20
            
            # Create a tighter dashboard
            dash_w = disp_w + sidebar_w + (padding * 3)
            dash_h = 650
            dashboard = np.zeros((dash_h, dash_w, 3), dtype=np.uint8)
            dashboard[:] = (25, 25, 25) # Slightly lighter dark gray

            # 1. Draw Board
            board_disp = cv2.resize(board_img, (disp_w, 600))
            # Highlight cell
            hl_x1, hl_y1 = int(x1 * scale), int(y1 * scale)
            hl_x2, hl_y2 = int(x2 * scale), int(y2 * scale)
            cv2.rectangle(board_disp, (hl_x1, hl_y1), (hl_x2, hl_y2), (0, 255, 255), 3)
            dashboard[padding:padding+600, padding:padding+disp_w] = board_disp
            
            # 2. Setup Sidebar Positioning
            x_side = disp_w + (padding * 2)
            
            # 3. Computer Vision View (Smaller thumbnail)
            zoom_size = 200
            zoom_disp = cv2.resize(zoom_view, (zoom_size, zoom_size))
            dashboard[padding+30:padding+30+zoom_size, x_side:x_side+zoom_size] = zoom_disp
            cv2.putText(dashboard, "CV VIEW", (x_side, padding+20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 2)

            # 4. Metrics Table
            stats_y = padding + zoom_size + 70
            cv2.putText(dashboard, "METRICS", (x_side, stats_y - 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            lines = [
                ("Blobs", f"{num_blobs}"),
                ("Circ", f"{metrics['Circularity']:.2f}"),
                ("Verts", f"{metrics['Vertices']}"),
                ("Solid", f"{metrics['Solidity']:.2f}"),
                ("Ext", f"{metrics['Extent']:.2f}")
            ]

            for i, (label_text, val_text) in enumerate(lines):
                y_pos = stats_y + (i * 30)
                # Label
                cv2.putText(dashboard, f"{label_text}:", (x_side, y_pos), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
                # Value (shifted right for alignment)
                cv2.putText(dashboard, val_text, (x_side + 100, y_pos), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # 5. Decision (The "Hero" element)
            dec_y = stats_y + 180
            cv2.rectangle(dashboard, (x_side - 5, dec_y - 30), (dash_w - 10, dec_y + 80), (40, 40, 40), -1)
            cv2.putText(dashboard, "DECISION", (x_side, dec_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            cv2.putText(dashboard, label.upper(), (x_side, dec_y + 45), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            cv2.putText(dashboard, reason, (x_side, dec_y + 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1)

            cv2.imshow("Act 2: Geometric Analysis of Shapes", dashboard)
            if cv2.waitKey(0) == 27: 
                cv2.destroyAllWindows()
                return
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_live_scanner(IMAGE_PATH)