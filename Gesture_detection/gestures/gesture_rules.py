from gestures.gesture_labels import GESTURE_LABELS

def fingers_up(hand_landmarks):
    tips = [4, 8, 12, 16, 20]
    fingers = []

    # Thumb
    fingers.append(
        hand_landmarks.landmark[4].x <
        hand_landmarks.landmark[3].x
    )

    # Other fingers
    for tip in tips[1:]:
        fingers.append(
            hand_landmarks.landmark[tip].y <
            hand_landmarks.landmark[tip - 2].y
        )

    return fingers.count(True)


def recognize_gesture(hand_landmarks):
    count = fingers_up(hand_landmarks)

    if count == 5:
        return GESTURE_LABELS["HI"]
    elif count == 0:
        return GESTURE_LABELS["BYE"]
    elif count == 2:
        return GESTURE_LABELS["VICTORY"]
    else:
        return GESTURE_LABELS["UNKNOWN"]
