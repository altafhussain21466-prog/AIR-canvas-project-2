import mediapipe as mp
import cv2

class HandTracker:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.results = None

    def find_hands(self, frame, draw=True):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(rgb)
        if draw and self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame, hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS
                )
        return frame

    def get_hand_position(self, frame, hand_number=0):
        landmark_list = []
        if self.results and self.results.multi_hand_landmarks:
            if len(self.results.multi_hand_landmarks) > hand_number:
                hand = self.results.multi_hand_landmarks[hand_number]
                h, w, _ = frame.shape
                for id, lm in enumerate(hand.landmark):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    landmark_list.append((id, cx, cy))
        return landmark_list

    def get_finger_position(self, frame, finger_id, hand_number=0):
        positions = self.get_hand_position(frame, hand_number)
        if not positions:
            return None
        for id, x, y in positions:
            if id == finger_id:
                return (x, y)
        return None

    def get_finger_up_status(self, frame, hand_number=0):
        positions = self.get_hand_position(frame, hand_number)
        if not positions:
            return [False] * 5
        landmarks = dict([(id, (x, y)) for id, x, y in positions])
        finger_tips = [8, 12, 16, 20]
        finger_pips = [6, 10, 14, 18]
        fingers_up = []
        if landmarks[4][0] > landmarks[3][0]:
            fingers_up.append(True)
        else:
            fingers_up.append(False)
        for tip, pip in zip(finger_tips, finger_pips):
            if landmarks[tip][1] < landmarks[pip][1]:
                fingers_up.append(True)
            else:
                fingers_up.append(False)
        return fingers_up