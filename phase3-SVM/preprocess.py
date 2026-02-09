# preprocess.py
import cv2
import numpy as np

def standardize_cell(img):
    """
    Turns any cell image (dark/light/messy) into a 
    perfectly centered, binary 32x32 icon.
    """
    # 1. Resize to 32x32 standard for processing
    img = cv2.resize(img, (64, 64))
    
    # 2. Blur slightly to remove jpeg noise
    blurred = cv2.GaussianBlur(img, (3, 3), 0)
    
    # 3. Corner Difference Strategy (The "Universal" Isolator)
    # Background color = average of the 4 corners
    corner_TL = blurred[0:3, 0:3]
    corner_TR = blurred[0:3, -3:]
    corner_BL = blurred[-3:, 0:3]
    corner_BR = blurred[-3:, -3:]
    
    corners = np.vstack((corner_TL, corner_TR, corner_BL, corner_BR))
    avg_bg_color = np.mean(corners, axis=(0, 1))
    
    # Calculate difference from background
    diff = cv2.absdiff(blurred, avg_bg_color.astype(np.uint8))
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    
    # 4. Threshold (Keep pixels that are >30 units different from BG)
    _, mask = cv2.threshold(diff_gray, 30, 255, cv2.THRESH_BINARY)
    
    # 5. Cleanup (Remove noise, fill holes)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2,2), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8))
    
    # 6. Safety Border (Delete edge noise)
    h, w = mask.shape
    cv2.rectangle(mask, (0,0), (w, h), 0, 2) # 2px black border around edge
    
    # ---------------------------------------------------------
    # 7. FINAL RE-CENTERING (The Polish)
    # ---------------------------------------------------------
    # Now that we have a clean shape, let's find it and center it exactly.
    coords = cv2.findNonZero(mask)
    
    # Create fresh black canvas
    final_img = np.zeros((64, 64), dtype=np.uint8)
    
    if coords is not None:
        # Get bounding box of the shape
        x, y, w, h = cv2.boundingRect(coords)
        
        # Crop just the shape
        shape = mask[y:y+h, x:x+w]
        
        # Calculate centered position
        center_x = (64 - w) // 2
        center_y = (64 - h) // 2
        
        # Paste it
        final_img[center_y:center_y+h, center_x:center_x+w] = shape
        
    return final_img