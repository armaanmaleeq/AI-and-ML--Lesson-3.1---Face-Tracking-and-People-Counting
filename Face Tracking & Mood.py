import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)

face_id = 1
prev_center = None


def distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    emotion = "Neutral"

    if results.multi_face_landmarks:
        for landmarks in results.multi_face_landmarks:
            points = landmarks.landmark

            xs = [int(p.x * w) for p in points]
            ys = [int(p.y * h) for p in points]

            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)

            center = ((x_min + x_max) // 2, (y_min + y_max) // 2)

            if prev_center is None:
                prev_center = center
            else:
                if distance(center, prev_center) > 50:
                    face_id += 1
                prev_center = center

            # Mouth landmarks
            left_mouth = (int(points[61].x * w), int(points[61].y * h))
            right_mouth = (int(points[291].x * w), int(points[291].y * h))
            top_lip = (int(points[13].x * w), int(points[13].y * h))
            bottom_lip = (int(points[14].x * w), int(points[14].y * h))

            mouth_width = distance(left_mouth, right_mouth)
            mouth_open = distance(top_lip, bottom_lip)

            # Eye landmarks
            left_eye_top = (int(points[159].x * w), int(points[159].y * h))
            left_eye_bottom = (int(points[145].x * w), int(points[145].y * h))
            eye_open = distance(left_eye_top, left_eye_bottom)

            # Emotion logic
            if mouth_open > 25 and eye_open > 6:
                emotion = "Surprised"
            elif mouth_width > 60:
                emotion = "Happy"
            elif mouth_open < 10:
                emotion = "Sad"
            else:
                emotion = "Neutral"

            # Draw face box
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            # Show ID
            cv2.putText(frame, f"ID: {face_id}", (x_min, y_min - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            # Show emotion
            cv2.putText(frame, emotion, (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("Emotion Detection and Face Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()