import cv2
from mediapipe import solutions
from apply_fiiltr import ApplyFiltr


def main():
    apply_mask_instance = ApplyFiltr("masks/anbu_mask.png")
    face_mesh = solutions.face_mesh.FaceMesh()
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        elif cv2.waitKey(1) & 0xFF == ord('w'):
            apply_mask_instance.filter_enabled = "BLACK_BARS"
        elif cv2.waitKey(1) & 0xFF == ord('e'):
            apply_mask_instance.filter_enabled = "APPLY_MASK"
        elif cv2.waitKey(1) & 0xFF == ord('r'):
            apply_mask_instance.filter_enabled = "SNOW"
        elif cv2.waitKey(1) & 0xFF == ord('t'):
            apply_mask_instance.filter_enabled = "SEPIA"
        elif cv2.waitKey(1) & 0xFF == ord('y'):
            apply_mask_instance.filter_enabled = "CARTOON"
        elif cv2.waitKey(1) & 0xFF == ord('u'):
            apply_mask_instance.filter_enabled = "BLUR"
        elif cv2.waitKey(1) & 0xFF == ord('i'):
            apply_mask_instance.filter_enabled = "EDGE_DETECTION"
        elif cv2.waitKey(1) & 0xFF == ord('o'):
            apply_mask_instance.filter_enabled = "INVERT"
        elif cv2.waitKey(1) & 0xFF == ord('z'):
            apply_mask_instance.filter_enabled = ""

        ret, frame = cap.read()
        if not ret:
            break
        
        results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        frame = apply_mask_instance.process_frame(frame, results)

        cv2.imshow('Face with Mask', frame)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()