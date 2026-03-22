import cv2

def draw_text(frame, text):
    cv2.putText(
        frame,
        f"Gesture: {text}",
        (30, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 255, 0),
        3
    )
