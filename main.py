import cv2
from mediapipe import solutions
import numpy as np
import math

filter_enabled = False
OFFSET_UP = 50

def main():
    global filter_enabled

    face_mesh = solutions.face_mesh.FaceMesh()
    cap = cv2.VideoCapture(0)

    mask = cv2.imread("maska.png", cv2.IMREAD_UNCHANGED)

    while cap.isOpened():
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        elif cv2.waitKey(1) & 0xFF == ord('w'):
            filter_enabled = True
        elif cv2.waitKey(1) & 0xFF == ord('z'):
            filter_enabled = False

        ret, frame = cap.read()
        if not ret:
            break

        if filter_enabled:
            frame = add_black_bars(frame)

        results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if results.multi_face_landmarks:
            frame_with_mask = apply_mask(frame, mask, results)
            cv2.imshow('Face with Mask', frame_with_mask)

    cap.release()
    cv2.destroyAllWindows()

def apply_mask(frame, mask, detection_result):
    frame_with_mask = np.copy(frame)

    for face_landmarks in detection_result.multi_face_landmarks:
        face_center_x = (face_landmarks.landmark[0].x + face_landmarks.landmark[1].x) / 2 * frame.shape[1]
        face_center_y = (face_landmarks.landmark[0].y + face_landmarks.landmark[1].y) / 2 * frame.shape[0]

        left_eye_x = face_landmarks.landmark[33].x * frame.shape[1]
        left_eye_y = face_landmarks.landmark[33].y * frame.shape[0]
        right_eye_x = face_landmarks.landmark[263].x * frame.shape[1]
        right_eye_y = face_landmarks.landmark[263].y * frame.shape[0]

        angle = math.degrees(math.atan2(right_eye_y - left_eye_y, right_eye_x - left_eye_x))
        angle = -angle

        x_offset = int(face_center_x - mask.shape[1] / 2)
        y_offset = int(face_center_y - mask.shape[0] / 2) - OFFSET_UP

        if mask.shape[2] == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2BGRA)

        center = (mask.shape[1] // 2, mask.shape[0] // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_mask = cv2.warpAffine(mask, rotation_matrix, (mask.shape[1], mask.shape[0]))

        frame_with_mask = overlay_transparent(frame_with_mask, rotated_mask, x_offset, y_offset)

    return frame_with_mask

def overlay_transparent(background, overlay, x, y):
    background_width = background.shape[1]
    background_height = background.shape[0]

    if x >= background_width or y >= background_height:
        return background

    h, w = overlay.shape[0], overlay.shape[1]

    if x + w > background_width:
        w = background_width - x
        overlay = overlay[:, :w]

    if y + h > background_height:
        h = background_height - y
        overlay = overlay[:h]

    if w <= 0 or h <= 0:
        return background

    overlay_image = overlay[..., :3]
    mask = overlay[..., 3:] / 255.0

    background[y:y+h, x:x+w] = (1.0 - mask) * background[y:y+h, x:x+w] + mask * overlay_image

    return background

def add_black_bars(frame):
    BAR_HEIGHT = 70
    height, width, _ = frame.shape
    black_bar = np.zeros((BAR_HEIGHT, width, 3), dtype=np.uint8)

    frame[:BAR_HEIGHT, :, :] = black_bar 
    frame[-BAR_HEIGHT:, :, :] = black_bar

    return frame

if __name__ == "__main__":
    main()
