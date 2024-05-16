import cv2
from mediapipe import solutions
import numpy as np
import matplotlib.pyplot as plt

BAR_HEIGHT = 70
filter_enabled = False

def main():
    global filter_enabled

    face_mesh = solutions.face_mesh.FaceMesh()
    cap = cv2.VideoCapture(0)

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
            cv2.imshow('Face Mesh', frame)

    cap.release()
    cv2.destroyAllWindows()


def draw_landmarks_on_image(rgb_image, detection_result):
    annotated_image = np.copy(rgb_image)

    if detection_result.multi_face_landmarks:
        for face_landmarks in detection_result.multi_face_landmarks:
            for landmark in face_landmarks.landmark:
                x = landmark.x * annotated_image.shape[1]
                y = landmark.y * annotated_image.shape[0]
                cv2.circle(annotated_image, (int(x), int(y)), 2, (0, 255, 0), -1)

    return annotated_image


def plot_face_blendshapes_bar_graph(face_blendshapes):
    face_blendshapes_names = [face_blendshapes_category.category_name for face_blendshapes_category in face_blendshapes]
    face_blendshapes_scores = [face_blendshapes_category.score for face_blendshapes_category in face_blendshapes]
    face_blendshapes_ranks = range(len(face_blendshapes_names))

    fig, ax = plt.subplots(figsize=(12, 12))
    bar = ax.barh(face_blendshapes_ranks, face_blendshapes_scores, label=[str(x) for x in face_blendshapes_ranks])
    ax.set_yticks(face_blendshapes_ranks, face_blendshapes_names)
    ax.invert_yaxis()

    for score, patch in zip(face_blendshapes_scores, bar.patches):
        plt.text(patch.get_x() + patch.get_width(), patch.get_y(), f"{score:.4f}", va="top")

    ax.set_xlabel('Score')
    ax.set_title("Face Blendshapes")
    plt.tight_layout()
    plt.show()


def add_black_bars(frame):
    height, width, _ = frame.shape
    black_bar = np.zeros((BAR_HEIGHT, width, 3), dtype=np.uint8)

    frame[:BAR_HEIGHT, :, :] = black_bar 
    frame[-BAR_HEIGHT:, :, :] = black_bar

    return frame

if __name__ == "__main__":
    main()
