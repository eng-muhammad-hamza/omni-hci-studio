"""
OmniHCI Studio - MediaPipe Lip Landmark & Vowel Recognition Engine
Recognizes visemes / vowel shapes: A, E, I, O, U, and closed mouth.
"""
import numpy as np
import cv2
from PIL import Image

OUTER_LIP = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
             291, 375, 321, 405, 314, 17, 84, 181, 91, 146]
INNER_LIP = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
             308, 324, 318, 402, 317, 14, 87, 178, 88, 95]

LIP_TOP_OUTER = 0
LIP_BOTTOM_OUTER = 17
LIP_LEFT_OUTER = 61
LIP_RIGHT_OUTER = 291
LIP_TOP_INNER = 13
LIP_BOTTOM_INNER = 14

def get_lip_points(face_landmarks, img_shape) -> dict:
    h, w = img_shape[:2]
    def pt(idx):
        lm = face_landmarks.landmark[idx]
        return np.array([lm.x * w, lm.y * h])

    return {
        "outer": np.array([pt(i) for i in OUTER_LIP]),
        "inner": np.array([pt(i) for i in INNER_LIP]),
        "top_outer": pt(LIP_TOP_OUTER),
        "bottom_outer": pt(LIP_BOTTOM_OUTER),
        "left": pt(LIP_LEFT_OUTER),
        "right": pt(LIP_RIGHT_OUTER),
        "top_inner": pt(LIP_TOP_INNER),
        "bottom_inner": pt(LIP_BOTTOM_INNER),
    }

def compute_lip_metrics(lip_pts: dict) -> dict:
    width = float(np.linalg.norm(lip_pts["right"] - lip_pts["left"]))
    outer_h = float(np.linalg.norm(lip_pts["bottom_outer"] - lip_pts["top_outer"]))
    inner_h = float(np.linalg.norm(lip_pts["bottom_inner"] - lip_pts["top_inner"]))
    mar = inner_h / (width + 1e-6)
    wh_ratio = width / (outer_h + 1e-6)
    open_ratio = inner_h / (outer_h + 1e-6)

    inner_area = cv2.contourArea(lip_pts["inner"].astype(np.float32).reshape(-1, 1, 2))
    inner_perimeter = cv2.arcLength(lip_pts["inner"].astype(np.float32).reshape(-1, 1, 2), True)
    roundness = float((4 * np.pi * inner_area) / (inner_perimeter ** 2 + 1e-6))

    return {
        "width": width,
        "outer_height": outer_h,
        "inner_height": inner_h,
        "mar": mar,
        "wh_ratio": wh_ratio,
        "open_ratio": open_ratio,
        "roundness": roundness
    }

def classify_vowel(m: dict) -> tuple:
    mar = m["mar"]
    wh = m["wh_ratio"]
    open_r = m["open_ratio"]
    roundness = m["roundness"]

    if mar < 0.08 or open_r < 0.12:
        return "Closed / Neutral", "Lips closed or resting"
    if mar > 0.40 and roundness > 0.45:
        return "A (as in Father)", "Wide vertical opening, high MAR"
    if roundness > 0.60 and mar > 0.25:
        return "O (as in Boat)", "Round circular aperture"
    if mar > 0.15 and roundness > 0.40 and wh < 2.5:
        return "U (as in Boot)", "Pursed, small rounded opening"
    if wh > 2.8 and mar < 0.28:
        return "I (as in Beet)", "Wide horizontal stretch, narrow opening"
    if wh > 2.2 and mar < 0.35:
        return "E (as in Bait)", "Moderate horizontal stretch"
    return "Open / Transition", "General vocal tract opening"

def process_lip_vowel(image_input) -> tuple:
    if image_input is None:
        return None, {}, "No image provided."

    if isinstance(image_input, Image.Image):
        img_np = np.array(image_input)
    else:
        img_np = image_input.copy()

    try:
        import mediapipe as mp
    except ImportError:
        return img_np, {}, "MediaPipe is not installed."

    mp_face_mesh = mp.solutions.face_mesh
    annotated = img_np.copy()

    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    ) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))

        if not results.multi_face_landmarks:
            return annotated, {"face_detected": False}, "No face detected in image."

        face_lms = results.multi_face_landmarks[0]
        lip_pts = get_lip_points(face_lms, img_np.shape)
        metrics = compute_lip_metrics(lip_pts)
        vowel, desc = classify_vowel(metrics)

        # Draw outer & inner contours
        outer_poly = lip_pts["outer"].astype(np.int32).reshape((-1, 1, 2))
        inner_poly = lip_pts["inner"].astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(annotated, [outer_poly], True, (0, 255, 255), 2)
        cv2.polylines(annotated, [inner_poly], True, (0, 0, 255), 2)

        # Draw label
        cv2.putText(annotated, f"Vowel: {vowel}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(annotated, f"MAR: {metrics['mar']:.3f} | Round: {metrics['roundness']:.3f}", (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)

        metrics["detected_vowel"] = vowel
        metrics["description"] = desc

        return annotated, metrics, f"Vowel Detected: {vowel} ({desc})"
