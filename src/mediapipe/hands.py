"""
OmniHCI Studio - MediaPipe Hand Gesture & Finger State Recognizer
"""
import numpy as np
import cv2
from PIL import Image

# Landmark indices
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

FINGER_TIPS = [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
FINGER_PIPS = [THUMB_IP, INDEX_PIP, MIDDLE_PIP, RING_PIP, PINKY_PIP]
FINGER_MCPS = [THUMB_MCP, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP]

def distance(p1: np.ndarray, p2: np.ndarray) -> float:
    return float(np.linalg.norm(p1[:2] - p2[:2]))

def is_finger_up(pts: np.ndarray, finger_idx: int, handedness: str = "Right") -> bool:
    tip = FINGER_TIPS[finger_idx]
    pip = FINGER_PIPS[finger_idx]
    if finger_idx == 0:  # Thumb
        mcp = THUMB_MCP
        if handedness == "Right":
            return pts[tip][0] < pts[mcp][0]
        else:
            return pts[tip][0] > pts[mcp][0]
    else:
        return pts[tip][1] < pts[pip][1]

def count_fingers(pts: np.ndarray, handedness: str = "Right") -> list:
    return [is_finger_up(pts, i, handedness) for i in range(5)]

def classify_gesture(pts: np.ndarray, handedness: str = "Right") -> tuple:
    fingers = count_fingers(pts, handedness)
    thumb, index, middle, ring, pinky = fingers
    n = sum(fingers)

    if n == 0:
        return "✊ Fist", fingers
    if n == 5:
        return "🖐 Open Hand", fingers
    if thumb and not index and not middle and not ring and not pinky:
        if pts[THUMB_TIP][1] < pts[WRIST][1]:
            return "👍 Thumbs Up", fingers
        else:
            return "👎 Thumbs Down", fingers
    if not thumb and index and not middle and not ring and not pinky:
        return "☝️ Pointing", fingers
    if not thumb and index and middle and not ring and not pinky:
        return "✌️ Peace", fingers
    if index and middle and ring and pinky and not thumb:
        d = distance(pts[THUMB_TIP], pts[INDEX_TIP])
        ref = distance(pts[INDEX_MCP], pts[PINKY_MCP])
        if d < ref * 0.35:
            return "👌 OK", fingers
    d_pinch = distance(pts[THUMB_TIP], pts[INDEX_TIP])
    ref = distance(pts[INDEX_MCP], pts[MIDDLE_MCP])
    if d_pinch < ref * 0.5:
        return "🤏 Pinch", fingers

    return f"🔢 {n} Fingers", fingers

def process_hand_gesture(image_input) -> tuple:
    """
    Process image using MediaPipe Hands.
    Returns: (annotated_image, metrics_dict, status_text)
    """
    if image_input is None:
        return None, {}, "No image provided."

    if isinstance(image_input, Image.Image):
        img_np = np.array(image_input)
    else:
        img_np = image_input.copy()

    try:
        import mediapipe as mp
    except ImportError:
        return img_np, {}, "MediaPipe is not installed. Please install mediapipe."

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles

    h, w = img_np.shape[:2]
    annotated = img_np.copy()

    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.5
    ) as hands:
        results = hands.process(cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))

        if not results.multi_hand_landmarks:
            return annotated, {"hands_detected": 0}, "No hands detected in image."

        metrics = {"hands_detected": len(results.multi_hand_landmarks), "details": []}
        
        for idx, hand_lms in enumerate(results.multi_hand_landmarks):
            handedness = "Right"
            if results.multi_handedness and idx < len(results.multi_handedness):
                handedness = results.multi_handedness[idx].classification[0].label

            pts = np.array([[lm.x * w, lm.y * h, lm.z * w] for lm in hand_lms.landmark], dtype=np.float32)
            gesture, fingers = classify_gesture(pts, handedness)
            
            mp_drawing.draw_landmarks(
                annotated,
                hand_lms,
                mp_hands.HAND_CONNECTIONS,
                mp_styles.get_default_hand_landmarks_style(),
                mp_styles.get_default_hand_connections_style()
            )

            # Draw label
            wrist_pt = (int(pts[WRIST][0]), int(pts[WRIST][1]))
            cv2.putText(annotated, f"{handedness}: {gesture}", (wrist_pt[0] - 20, max(25, wrist_pt[1] - 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

            metrics["details"].append({
                "hand_index": idx + 1,
                "handedness": handedness,
                "gesture": gesture,
                "fingers_up_count": sum(fingers),
                "finger_states": {
                    "thumb": fingers[0],
                    "index": fingers[1],
                    "middle": fingers[2],
                    "ring": fingers[3],
                    "pinky": fingers[4]
                }
            })

        summary = f"Detected {len(results.multi_hand_landmarks)} hand(s): " + ", ".join([f"{d['handedness']} ({d['gesture']})" for d in metrics["details"]])
        return annotated, metrics, summary
