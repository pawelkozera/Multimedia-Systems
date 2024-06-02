import cv2
from mediapipe import solutions
from apply_fiiltr import ApplyFiltr
from tkinter import *
from PIL import Image, ImageTk

class FaceFilterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Filter App")
        self.root.geometry("800x500")
        
        self.canvas = Canvas(self.root, width=700, height=600)
        self.canvas.pack(side=LEFT)
        
        self.button_frame = Frame(self.root)
        self.button_frame.pack(side=LEFT, fill=BOTH, expand=True)

        self.create_buttons()

        self.apply_mask_instance = ApplyFiltr()
        self.face_mesh = solutions.face_mesh.FaceMesh()
        self.cap = cv2.VideoCapture(0)

        self.update_frame()

    def create_buttons(self):
        button_texts = [
            ('Apply Mask', self.apply_mask),
            ('Clear Mask', self.clear_mask),
            ('Black Bars', lambda: self.apply_filter("BLACK_BARS")),
            ('Snow', lambda: self.apply_filter("SNOW")),
            ('Sepia', lambda: self.apply_filter("SEPIA")),
            ('Cartoon', lambda: self.apply_filter("CARTOON")),
            ('Blur', lambda: self.apply_filter("BLUR")),
            ('Edge Detection', lambda: self.apply_filter("EDGE_DETECTION")),
            ('Invert', lambda: self.apply_filter("INVERT")),
            ('Clear Filter', self.clear_filter),
            ('Quit', self.quit_app)
        ]

        for text, command in button_texts:
            button = Button(self.button_frame, text=text, command=command)
            button.pack(pady=5)

    def apply_mask(self):
        self.apply_mask_instance.mask_enabled = "APPLY_MASK"
        self.apply_mask_instance.change_mask_index()

    def clear_mask(self):
        self.apply_mask_instance.mask_enabled = ""

    def apply_filter(self, filter_name):
        self.apply_mask_instance.filter_enabled = filter_name

    def clear_filter(self):
        self.apply_mask_instance.filter_enabled = ""

    def quit_app(self):
        self.cap.release()
        self.root.quit()

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            results = self.face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frame = self.apply_mask_instance.process_frame(frame, results)
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)
            self.canvas.create_image(0, 0, anchor=NW, image=imgtk)
            self.canvas.imgtk = imgtk
        
        self.root.after(10, self.update_frame)

if __name__ == "__main__":
    root = Tk()
    app = FaceFilterApp(root)
    root.mainloop()
