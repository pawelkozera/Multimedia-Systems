import cv2
import numpy as np
import math
import random


class ApplyFiltr:
    OFFSET_UP = 50

    def __init__(self, mask_path):
        self.mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        self.filter_enabled = ""

    def process_frame(self, frame, detection_result):
        if self.filter_enabled == "BLACK_BARS":
            return self.add_black_bars(frame)
        elif self.filter_enabled == "APPLY_MASK" and detection_result.multi_face_landmarks:
            return self.apply_mask(frame, detection_result)
        elif self.filter_enabled == "SNOW":
            return self.apply_snow_effect(frame)
        return frame

    def apply_mask(self, frame, detection_result):
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

            x_offset = int(face_center_x - self.mask.shape[1] / 2)
            y_offset = int(face_center_y - self.mask.shape[0] / 2) - self.OFFSET_UP

            if self.mask.shape[2] == 3:
                self.mask = cv2.cvtColor(self.mask, cv2.COLOR_BGR2BGRA)

            center = (self.mask.shape[1] // 2, self.mask.shape[0] // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated_mask = cv2.warpAffine(self.mask, rotation_matrix, (self.mask.shape[1], self.mask.shape[0]))

            frame_with_mask = self.overlay_transparent(frame_with_mask, rotated_mask, x_offset, y_offset)

        return frame_with_mask

    def overlay_transparent(self, background, overlay, x, y):
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

    def add_black_bars(self, frame):
        BAR_HEIGHT = 70
        height, width, _ = frame.shape
        black_bar = np.zeros((BAR_HEIGHT, width, 3), dtype=np.uint8)

        frame[:BAR_HEIGHT, :, :] = black_bar 
        frame[-BAR_HEIGHT:, :, :] = black_bar

        return frame

    def apply_snow_effect(self, frame):
        snow_frame = np.copy(frame)
        snow_intensity = 0.02
        snowflake_size = 4

        num_snowflakes = int(snow_intensity * frame.shape[0] * frame.shape[1])
        for _ in range(num_snowflakes):
            x = random.randint(0, frame.shape[1] - 1)
            y = random.randint(0, frame.shape[0] - 1)
            color = (0, 255, 0) if random.random() < 0.5 else (100, 0, 255)
            cv2.circle(snow_frame, (x, y), snowflake_size, color, -1)

        return cv2.addWeighted(frame, 0.9, snow_frame, 0.1, 0)
