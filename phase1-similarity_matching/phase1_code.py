import cv2
import numpy as np
import os

def debug_ultimate_solver(screenshot_path, template_folder, grid_size=5):
    # --- 1. SETUP ---
    img_rgb = cv2.imread(screenshot_path)
    if img_rgb is None:
        print(f"Error: Could not load {screenshot_path}")
        return
    
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    h_main, w_main = img_gray.shape
    
    # Grid Calculation
    cell_h, cell_w = h_main // grid_size, w_main // grid_size
    
    # Prepare Visuals
    locked_img = img_rgb.copy()
    virtual_grid = [["." for _ in range(grid_size)] for _ in range(grid_size)]

    templates = []
    for f in os.listdir(template_folder):
        if f.lower().endswith(('.png', '.jpg')):
            templates.append((f.replace(".png", ""), cv2.imread(os.path.join(template_folder, f), 0)))

    print("Controls: [SPACE] Next Scale | [ESC] Quit")

    # --- 2. MAIN LOOP ---
    for label, raw_template in templates:
        # Loop scales relative to the CELL SIZE
        scales = np.linspace(0.8, 1.1, 8) 
        
        for scale in scales:
            target_w = int(cell_w * scale)
            target_h = int(cell_h * scale)
            
            if target_w >= w_main or target_h >= h_main: continue
            
            curr_template = cv2.resize(raw_template, (target_w, target_h))
            
            # A. MATCHING
            res = cv2.matchTemplate(img_gray, curr_template, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            
            # --- FIX 1: HIGH-CONTRAST HEATMAP ---
            # Clip negative values (noise) to 0
            res = np.maximum(res, 0)
            
            # Stretch contrast: 0.6 -> 1.0 becomes 0 -> 255
            contrast_floor = 0.6
            heatmap_stretched = (res - contrast_floor) / (1.0 - contrast_floor)
            heatmap_stretched = np.clip(heatmap_stretched, 0, 1)
            heatmap_uint8 = (heatmap_stretched * 255).astype(np.uint8)
            
            heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

            # --- FIX 2: CENTERED PADDING (Fixed Alignment) ---
            # We want the "hotspot" to appear at the CENTER of the object, 
            # not the top-left corner. So we shift the image by half the template size.
            
            h_res, w_res, _ = heatmap_color.shape
            
            # Calculate padding to center the result
            pad_top = target_h // 2
            pad_left = target_w // 2
            pad_bottom = h_main - h_res - pad_top
            pad_right = w_main - w_res - pad_left
            
            # Apply padding to all 4 sides
            # Using BORDER_CONSTANT (Black) helps visualize the "unsearchable" border area
            heatmap_display = cv2.copyMakeBorder(
                heatmap_color, 
                pad_top, pad_bottom, pad_left, pad_right, 
                cv2.BORDER_CONSTANT, value=(0, 0, 0)
            )   

            # Add Grid Lines
            for i in range(1, grid_size):
                cv2.line(heatmap_display, (i*cell_w, 0), (i*cell_w, h_main), (50, 50, 50), 1)
                cv2.line(heatmap_display, (0, i*cell_h), (w_main, i*cell_h), (50, 50, 50), 1)

            # --- SOLVER VIEW ---
            display_img = locked_img.copy()
            
            # Only draw boxes if score is genuinely high (User requested 0.90)
            threshold = 0.90
            loc = np.where(res >= threshold)
            
            candidates = []
            for pt in zip(*loc[::-1]):
                candidates.append(pt)
                cv2.rectangle(display_img, pt, (pt[0] + target_w, pt[1] + target_h), (255, 100, 0), 2)

            # UI Text
            info = f"{label} | Scale: {scale:.2f}x | Score: {max_val:.2f}"
            cv2.putText(display_img, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 4)
            cv2.putText(display_img, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            
            # SHOW WINDOWS
            cv2.imshow("1. Heatmap (Aligned)", heatmap_display)
            cv2.imshow("2. Solver View", display_img)
            
            # LOCK IN LOGIC
            if max_val > 0.90:
                for pt in candidates:
                    center_x, center_y = pt[0] + target_w//2, pt[1] + target_h//2
                    c, r = center_x // cell_w, center_y // cell_h
                    if 0 <= c < grid_size and 0 <= r < grid_size:
                        if virtual_grid[r][c] == ".":
                            virtual_grid[r][c] = label[0].upper()
                            cv2.rectangle(locked_img, pt, (pt[0] + target_w, pt[1] + target_h), (0, 255, 0), 2)

            key = cv2.waitKey(0)
            if key == 27: # ESC
                cv2.destroyAllWindows()
                return

    cv2.destroyAllWindows()
    print("Final Grid:")
    for r in virtual_grid: print(r)
# debug_portfolio_visualizer("Example1.png", "templates/", grid_size=4)



shapes_template_location = r"C:\Github Code\gap-solver-app\phase1-similarity_matching\assets\templates"
screenshot_filelocation = r"C:\Github Code\gap-solver-app\examples\5\Screenshot 2026-02-04 142421.png"             
debug_ultimate_solver(screenshot_filelocation, shapes_template_location)             
#cv2.imwrite("/Users/jordanchong/Desktop/github code/gap-solver-app/phase 1 image matching/assets/act1_failure.png", result)
