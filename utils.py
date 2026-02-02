import cv2
import numpy as np
import os

def load_gesture_images(folder_path):
    """
    Load specific monkey images from images/ folder into dictionary.
    Required files: default.jpg, think.jpg, mad_foldedarms.jpg, idea.jpg
    Returns:
        dict: Mapping gesture names to loaded images (numpy arrays).
    """
    images = {}
    
    # Map Gesture Name -> Filename
    gesture_files = {
        "Default": "default.jpg",
        "Think": "think.jpg",
        "Mad": "mad_foldedarms.jpg",
        "Idea": "idea.jpg",
        "Scared": "scared.jpg"
    }
    
    if not os.path.exists(folder_path):
        print(f"Error: Image folder '{folder_path}' does not exist.")
        return images

    for gesture, filename in gesture_files.items():
        path = os.path.join(folder_path, filename)
        if os.path.exists(path):
            try:
                # Load image
                img = cv2.imread(path)
                if img is not None:
                    # We will resize these dynamically in main.py to fit the split screen, 
                    # or resize here to a fixed size? 
                    # Let's keep them original size here and resize in main.py to match window height.
                    images[gesture] = img
                else:
                    print(f"Warning: Could not load {filename}")
            except Exception as e:
                print(f"Error loading {filename}: {e}")
        else:
            print(f"Warning: File {filename} not found in {folder_path}")
            
    return images

def calculate_finger_angle(a, b, c):
    """
    Calculates angle ABC (at point b).
    Args:
        a, b, c: [x, y] coordinates (or landmarks).
    Returns:
        angle: Angle in degrees.
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)
