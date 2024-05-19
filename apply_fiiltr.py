import cv2
import numpy as np
import math
import random


class ApplyFiltr:
    OFFSET_UP = 50

    def __init__(self):
        self.masks = [cv2.imread("masks/kakashi_mask.png", cv2.IMREAD_UNCHANGED), cv2.imread("masks/anbu_mask.png", cv2.IMREAD_UNCHANGED)]
        self.mask_index = 1
        self.mask = self.masks[0]
        self.filter_enabled = ""
        self.mask_enabled = ""

    def process_frame(self, frame, detection_result):
        if self.mask_enabled and detection_result.multi_face_landmarks:
            frame = self.apply_mask(frame, detection_result)

        if self.filter_enabled == "BLACK_BARS":
            return self.add_black_bars(frame)
        elif self.filter_enabled == "SNOW":
            return self.apply_snow_effect(frame)
        elif self.filter_enabled == "SEPIA":
            return self.apply_sepia(frame)
        elif self.filter_enabled == "CARTOON":
            return self.apply_cartoon(frame)
        elif self.filter_enabled == "BLUR":
            return self.apply_blur(frame)
        elif self.filter_enabled == "EDGE_DETECTION":
            return self.apply_edge_detection(frame)
        elif self.filter_enabled == "INVERT":
            return self.apply_invert(frame)
        
        return frame

    def change_mask_index(self):
        if self.mask_index >= len(self.masks) - 1:
            self.mask_index = 0
        else:
            self.mask_index += 1
        
        self.mask = self.masks[self.mask_index]

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

    def apply_sepia(self, frame):
        sepia_filter = np.array([[0.272, 0.534, 0.131],
                                [0.349, 0.686, 0.168],
                                [0.393, 0.769, 0.189]])
        sepia_frame = cv2.transform(frame, sepia_filter)
        sepia_frame = np.clip(sepia_frame, 0, 255).astype(np.uint8)

        return sepia_frame

    def apply_cartoon(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_blurred = cv2.medianBlur(gray, 7)
        edges = cv2.Canny(gray_blurred, 50, 150)
        adaptive_edges = cv2.adaptiveThreshold(gray_blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 2)
        combined_edges = cv2.bitwise_or(edges, adaptive_edges)
        color = cv2.bilateralFilter(frame, 9, 300, 300)
        cartoon = cv2.bitwise_and(color, color, mask=combined_edges)
        
        return cartoon

    def apply_blur(self, frame):
        return cv2.GaussianBlur(frame, (15, 15), 0)

    def apply_edge_detection(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)

        return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    def apply_invert(self, frame):
        return cv2.bitwise_not(frame)
