import cv2
from utils.hand_detector import HandDetector
from gestures.gesture_rules import recognize_gesture
from utils.draw_utils import draw_text

cap = cv2.VideoCapture(0)
detector = HandDetector()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    hand_landmarks = detector.detect_hands(frame)

    gesture_text = "No Hand"

    if hand_landmarks:
        gesture_text = recognize_gesture(hand_landmarks)

    draw_text(frame, gesture_text)

    cv2.imshow("Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
