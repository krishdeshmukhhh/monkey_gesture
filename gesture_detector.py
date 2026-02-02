import numpy as np

class GestureDetector:
    def __init__(self, min_confidence=0.7):
        self.min_confidence = min_confidence

    def _is_finger_extended(self, lm, finger_tip_idx, finger_pip_idx):
        # Universal check: Tip Y < PIP Y (assuming upright hand)
        # However, for "Think", finger might be pointed somewhat backwards or sideways if tilted.
        # So we also check Euclidean distance from Wrist? No, simpler.
        # Just check if Tip is NOT curled deeply. 
        # Tip to MCP distance > PIP to MCP distance?
        # Let's stick to the prompt's implied simple logic but relax it for "Think".
        return lm[finger_tip_idx].y < lm[finger_pip_idx].y

    def detect_gesture(self, multi_hand_landmarks, face_landmarks=None):
        """
        Analyze logic to return gesture name.
        Args:
            multi_hand_landmarks: List of Hand landmarks.
            face_landmarks: Single Face landmarks (if detected).
        """
        if not multi_hand_landmarks:
            return None

        # 1. 2-Hand Gestures (Mad vs Scared)
        if len(multi_hand_landmarks) == 2:
            h1 = multi_hand_landmarks[0].landmark
            h2 = multi_hand_landmarks[1].landmark
            
            # Average Height of Wrists
            avg_y = (h1[0].y + h2[0].y) / 2.0
            
            # Logic A: High Hands (Scared)
            # If hands are in upper half (e.g., < 0.55), assume Scared.
            # This covers both "Clasped near face" and "Hands on cheeks on sides of face".
            # We don't enforce close wrists for Scared anymore to allow "Home Alone" style.
            if avg_y < 0.55:
                # Optional: maybe check if palms are open or facing camera? 
                # For now, 2 hands high is a strong enough signal for Scared.
                return "Scared"
            
            # Logic B: Low Hands (Mad)
            # If hands are low, we require them to be crossed/close (Folded Arms).
            wrists_dist = np.sqrt((h1[0].x - h2[0].x)**2 + (h1[0].y - h2[0].y)**2)
            
            if wrists_dist < 0.45: # Generous threshold for folded arms
                return "Mad"

        # 2. Single Hand Gestures (Idea vs Think)
        for hand_landmarks in multi_hand_landmarks:
            lm = hand_landmarks.landmark
            
            # Index Extended?
            # Relaxed check for Think: Tip dist from MCP is large enough
            index_extended = self._is_finger_extended(lm, 8, 6)
            
            # Others Curled?
            middle_curled = not self._is_finger_extended(lm, 12, 10)
            ring_curled = not self._is_finger_extended(lm, 16, 14)
            pinky_curled = not self._is_finger_extended(lm, 20, 18)
            
            if index_extended and middle_curled and ring_curled and pinky_curled:
                # Potential Candidate for Idea or Think
                
                # --- FACE PROXIMITY CHECK (Prioritize this for Think) ---
                if face_landmarks:
                    # Mouth Center: Upper Lip (13) + Lower Lip (14) average
                    # Or just use Lower Lip (14)
                    mouth_x = face_landmarks.landmark[13].x
                    mouth_y = face_landmarks.landmark[13].y
                    
                    finger_tip_x = lm[8].x
                    finger_tip_y = lm[8].y
                    
                    dist_to_mouth = np.sqrt((finger_tip_x - mouth_x)**2 + (finger_tip_y - mouth_y)**2)
                    
                    # If very close to mouth, IT IS THINKING.
                    if dist_to_mouth < 0.15: # 15% of screen width/height? 
                        return "Think"
                        
                # --- FALLBACK / IDEA Logic ---
                
                # If not close to mouth, is it Idea?
                # Idea requires strict Verticality and Thumb Tuck.
                
                # Vertical check
                x_diff = abs(lm[8].x - lm[5].x)
                is_vertical = x_diff < 0.12 # Strict Vertical
                
                # Thumb check
                thumb_tip = lm[4]
                middle_mcp = lm[9]
                thumb_dist = np.sqrt((thumb_tip.x - middle_mcp.x)**2 + (thumb_tip.y - middle_mcp.y)**2)
                is_thumb_curled = thumb_dist < 0.2
                
                if is_vertical and is_thumb_curled:
                    return "Idea"
                    
                # If it wasn't vertical enough for Idea, and we didn't match face...
                # Maybe fallback to Think if high enough but no face detected?
                # Or just return None to avoid bugs.
                # User complaint: "Idea assumes I am doing think sometimes".
                # So we should be conservative with Think if no face is visible.
                
                if not face_landmarks:
                     # Fallback logic without face mesh
                     if lm[8].y < 0.35 and not is_vertical: # High but tilted?
                         return "Think"

        return None
