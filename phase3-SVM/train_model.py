import os
import cv2
import numpy as np
import joblib
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report
from preprocess import standardize_cell 

DATA_DIR = "dataset"
MODEL_FILE = "model.pkl"

def load_data():
    data = []
    labels = []
    classes = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d)) and d != "unsorted"]
    classes.sort()
    
    print(f"Found Classes: {classes}")
    
    for label in classes:
        path = os.path.join(DATA_DIR, label)
        print(f"Loading {label}...")
        for f in os.listdir(path):
            try:
                img = cv2.imread(os.path.join(path, f))
                if img is None: continue
                
                # Preprocess (will now use 64x64 if you updated preprocess.py)
                processed = standardize_cell(img)
                data.append(processed.flatten())
                labels.append(label)
            except: pass
                
    return np.array(data), np.array(labels), classes

# 1. Load Data
X, y, class_names = load_data()
X = X / 255.0  # Normalize

# 2. Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. DEFINE THE GRID (The "Fuzziness" Tuner)
# 'C': High C = stricter rules (less mistakes allowed)
# 'gamma': High gamma = stricter shape matching (looks for corners)
param_grid = {
    'C': [1, 10, 100, 1000],
    'gamma': [0.001, 0.01, 0.1, 'scale'],
    'kernel': ['rbf']
}

print("🔍 Tuning 'Fuzziness' (Grid Search)... this may take a moment...")
grid = GridSearchCV(SVC(class_weight='balanced'), param_grid, refit=True, verbose=2, cv=3)
grid.fit(X_train, y_train)


# 4. Report Results
print(f"✅ Best Parameters Found: {grid.best_params_}")
preds = grid.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, preds))

# 5. Save the Best Brain
joblib.dump({'model': grid.best_estimator_, 'classes': class_names}, MODEL_FILE)
print(f"Optimized model saved to {MODEL_FILE}")