import cv2
import mediapipe as mp
import time
import numpy as np
import sys
from gesture_detector import GestureDetector
from utils import load_gesture_images

def main():
    # --- Initialization ---
    
    # 1. Camera Setup
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not access webcam.")
        sys.exit(1)
    
    # Target 30 FPS for performance
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # Standard resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # 2. MediaPipe Setup
    mp_hands = mp.solutions.hands
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    
    hands = mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )
    
    # Optimization: refine_landmarks=False is much faster
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=False, 
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    # 3. Load Assets
    print("Loading gesture images...")
    gesture_images = load_gesture_images("images")
    if not gesture_images:
        print("Warning: No images loaded.")
    # Debug print loaded keys
    print(f"Loaded: {list(gesture_images.keys())}")
        
    # Pre-process images
    TARGET_HEIGHT = 720
    processed_images = {}
    
    for name, img in gesture_images.items():
        h, w = img.shape[:2]
        scale = TARGET_HEIGHT / h
        new_w = w # Don't scale up aggressively if not needed, but keeping logic same
        new_w = int(w * scale)
        resized = cv2.resize(img, (new_w, TARGET_HEIGHT))
        processed_images[name] = resized
        
    if "Default" not in processed_images:
        processed_images["Default"] = np.zeros((TARGET_HEIGHT, 500, 3), dtype=np.uint8)

    # 4. Logic/State Variables
    detector = GestureDetector(min_confidence=0.7)
    
    print("Starting... Press 'q' to quit.")
    
    prev_time = 0
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue

        # Flip frame (Mirror mode)
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        
        # Ensure frame is 720p
        if h != TARGET_HEIGHT:
            frame = cv2.resize(frame, (int(w * (TARGET_HEIGHT / h)), TARGET_HEIGHT))
            h, w, _ = frame.shape
            
        # Convert for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process
        results_hands = hands.process(frame_rgb)
        results_face = face_mesh.process(frame_rgb)
        
        # --- Visual Debug / Hitboxes ---
        
        # Draw Scared Threshold Line (Y = 0.55)
        thresh_y = int(h * 0.55)
        cv2.line(frame, (0, thresh_y), (w, thresh_y), (0, 0, 255), 1)
        cv2.putText(frame, "High/Low Line", (10, thresh_y - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    
        # Get Face Landmarks
        face_landmarks = None
        if results_face.multi_face_landmarks:
            face_landmarks = results_face.multi_face_landmarks[0]
            
            # Draw Mouth "Hitbox" (Landmark 13 and 14 are inner lips)
            # 13 is upper lip, 14 is lower lip
            lip_u = face_landmarks.landmark[13]
            lip_l = face_landmarks.landmark[14]
            
            cx, cy = int(lip_u.x * w), int(lip_u.y * h)
            
            # Draw Circle around mouth
            cv2.circle(frame, (cx, cy), 40, (255, 0, 255), 2) # 40px radius target
            cv2.putText(frame, "Mouth Target", (cx + 45, cy), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

        detected_gesture = "Default"
        
        if results_hands.multi_hand_landmarks:
            for hand_landmarks in results_hands.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1)
                )
                
                # Draw Index Finger Tip Hitbox
                idx_tip = hand_landmarks.landmark[8]
                tx, ty = int(idx_tip.x * w), int(idx_tip.y * h)
                cv2.circle(frame, (tx, ty), 10, (0, 255, 255), -1) # Yellow dot
            
            # Detect
            res = detector.detect_gesture(results_hands.multi_hand_landmarks, face_landmarks)
            if res:
                detected_gesture = res
        
        # --- UI Construction ---
        
        # Get right-side image
        right_img = processed_images.get(detected_gesture, processed_images["Default"])
        
        # Combine Canvas
        # Frame (Left) + Right Image (Right)
        combined_width = frame.shape[1] + right_img.shape[1]
        canvas = np.zeros((TARGET_HEIGHT, combined_width, 3), dtype=np.uint8)
        
        # Place Left
        canvas[:, :frame.shape[1]] = frame
        # Place Right
        canvas[:, frame.shape[1]:] = right_img
        
        # FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
        prev_time = curr_time
        
        cv2.putText(canvas, f"FPS: {int(fps)}", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.putText(canvas, f"Gesture: {detected_gesture}", (20, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # Show
        cv2.imshow('Gesture Detection', canvas)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
