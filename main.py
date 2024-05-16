import cv2
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import matplotlib.pyplot as plt

def main():
    # Utwórz obiekt do wykrywania twarzy
    face_mesh = solutions.face_mesh.FaceMesh()

    # Uruchom kamerę
    cap = cv2.VideoCapture(0)  # Numer kamery, 0 oznacza domyślną kamerę

    while cap.isOpened():
        # Wczytaj ramkę z kamery
        ret, frame = cap.read()
        if not ret:
            break

        # Przetwórz ramkę - wykrywanie punktów charakterystycznych twarzy
        results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if results.multi_face_landmarks:
            # Wywołaj funkcję do rysowania punktów charakterystycznych twarzy
            #annotated_frame = draw_landmarks_on_image(frame, results)

            # Wyświetl zaktualizowaną ramkę
            cv2.imshow('Face Mesh', frame)

        # Wyjście z pętli po naciśnięciu klawisza 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Zwolnij obiekt kamery i zakończ okno
    cap.release()
    cv2.destroyAllWindows()


def draw_landmarks_on_image(rgb_image, detection_result):
    annotated_image = np.copy(rgb_image)

    # Sprawdź, czy wykryto jakieś twarze
    if detection_result.multi_face_landmarks:
        # Pętla po wykrytych twarzach
        for face_landmarks in detection_result.multi_face_landmarks:
            # Rysowanie punktów charakterystycznych twarzy
            for landmark in face_landmarks.landmark:
                # Pobierz współrzędne punktu charakterystycznego
                x = landmark.x * annotated_image.shape[1]
                y = landmark.y * annotated_image.shape[0]

                # Narysuj punkt na obrazie
                cv2.circle(annotated_image, (int(x), int(y)), 2, (0, 255, 0), -1)

    return annotated_image


def plot_face_blendshapes_bar_graph(face_blendshapes):
    # Extract the face blendshapes category names and scores.
    face_blendshapes_names = [face_blendshapes_category.category_name for face_blendshapes_category in face_blendshapes]
    face_blendshapes_scores = [face_blendshapes_category.score for face_blendshapes_category in face_blendshapes]
    # The blendshapes are ordered in decreasing score value.
    face_blendshapes_ranks = range(len(face_blendshapes_names))

    fig, ax = plt.subplots(figsize=(12, 12))
    bar = ax.barh(face_blendshapes_ranks, face_blendshapes_scores, label=[str(x) for x in face_blendshapes_ranks])
    ax.set_yticks(face_blendshapes_ranks, face_blendshapes_names)
    ax.invert_yaxis()

    # Label each bar with values
    for score, patch in zip(face_blendshapes_scores, bar.patches):
        plt.text(patch.get_x() + patch.get_width(), patch.get_y(), f"{score:.4f}", va="top")

    ax.set_xlabel('Score')
    ax.set_title("Face Blendshapes")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()