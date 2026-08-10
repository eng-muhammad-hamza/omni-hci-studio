"""
OmniHCI Studio - MediaPipe Face Mesh, EAR (Drowsiness), MAR (Yawn), & Emotion Perception
"""
import numpy as np
import cv2
from PIL import Image

LEFT_EAR_PTS = [362, 385, 387, 263, 373, 380]
RIGHT_EAR_PTS = [33, 160, 158, 133, 153, 144]
MAR_PTS = [78, 308, 82, 312, 14, 317, 87, 402]

def get_pts(face_landmarks, indices, img_shape) -> np.ndarray:
    h, w = img_shape[:2]
    return np.array([[face_landmarks.landmark[i].x * w,
                      face_landmarks.landmark[i].y * h]
                     for i in indices], dtype=np.float32)

def compute_ear(eye_pts: np.ndarray) -> float:
    """
    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    """
    a = np.linalg.norm(eye_pts[1] - eye_pts[5])
    b = np.linalg.norm(eye_pts[2] - eye_pts[4])
    c = np.linalg.norm(eye_pts[0] - eye_pts[3])
    return float((a + b) / (2.0 * c + 1e-6))

def compute_mar(mouth_pts: np.ndarray) -> float:
    """
    MAR = (||p2-p8|| + ||p3-p7|| + ||p4-p6||) / (3 * ||p1-p5||)
    """
    a = np.linalg.norm(mouth_pts[2] - mouth_pts[7])
    b = np.linalg.norm(mouth_pts[3] - mouth_pts[6])
    c = np.linalg.norm(mouth_pts[4] - mouth_pts[5])
    d = np.linalg.norm(mouth_pts[0] - mouth_pts[1])
    return float((a + b + c) / (3.0 * d + 1e-6))

def estimate_emotion(ear: float, mar: float, brow_dist: float = 0.0) -> str:
    """Affective state estimation combining EAR, MAR, and facial geometry."""
    if mar > 0.65:
        return "😲 Surprised / Yawning"
    elif ear < 0.20:
        return "😴 Drowsy / Closed Eyes"
    elif mar > 0.35:
        return "😄 Happy / Smiling"
    else:
        return "😐 Neutral / Attentive"

def process_face_mesh(image_input) -> tuple:
    """
    Process image using MediaPipe Face Mesh.
    Returns: (annotated_image, metrics_dict, summary_text)
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
        return img_np, {}, "MediaPipe is not installed."

    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    h, w = img_np.shape[:2]
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

        # Draw mesh tessellation and irises
        mp_drawing.draw_landmarks(
            image=annotated,
            landmark_list=face_lms,
            connections=mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
        )
        mp_drawing.draw_landmarks(
            image=annotated,
            landmark_list=face_lms,
            connections=mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
        )

        left_eye = get_pts(face_lms, LEFT_EAR_PTS, img_np.shape)
        right_eye = get_pts(face_lms, RIGHT_EAR_PTS, img_np.shape)
        mouth = get_pts(face_lms, MAR_PTS, img_np.shape)

        left_ear = compute_ear(left_eye)
        right_ear = compute_ear(right_eye)
        avg_ear = float((left_ear + right_ear) / 2.0)
        mar = compute_mar(mouth)

        emotion = estimate_emotion(avg_ear, mar)
        drowsy_state = "⚠️ DROWSINESS DETECTED" if avg_ear < 0.20 else "✅ Alert"
        yawn_state = "⚠️ YAWNING DETECTED" if mar > 0.65 else "✅ Normal Mouth"

        # Overlay metrics on frame
        cv2.putText(annotated, f"EAR: {avg_ear:.3f} ({drowsy_state})", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(annotated, f"MAR: {mar:.3f} ({yawn_state})", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(annotated, f"Emotion: {emotion}", (20, 105),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        metrics = {
            "face_detected": True,
            "left_ear": round(left_ear, 4),
            "right_ear": round(right_ear, 4),
            "avg_ear": round(avg_ear, 4),
            "mar": round(mar, 4),
            "drowsiness": drowsy_state,
            "yawning": yawn_state,
            "emotion": emotion
        }

        summary = f"EAR: {avg_ear:.3f} | MAR: {mar:.3f} | State: {emotion}"
        return annotated, metrics, summary
