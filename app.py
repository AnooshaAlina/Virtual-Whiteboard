import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import cv2
import numpy as np
import mediapipe as mp
from collections import deque

# --- Global Config ---
st.set_page_config(page_title="AI Virtual Whiteboard", layout="wide")
st.title("🖐️ AI Virtual Whiteboard")

# Initialize Mediapipe outside the class to prevent re-initialization errors
mp_hands = mp.solutions.hands
hands_detector = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

class MyVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.bpoints = [deque(maxlen=1024)]
        self.gpoints = [deque(maxlen=1024)]
        self.rpoints = [deque(maxlen=1024)]
        self.ypoints = [deque(maxlen=1024)]
        
        self.blue_index = 0
        self.green_index = 0
        self.red_index = 0
        self.yellow_index = 0
        
        self.colorIndex = 0
        self.colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]

    def recv(self, frame):
        # Use .recv() instead of .transform() in newer streamlit-webrtc
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        framergb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Draw UI Buttons
        img = cv2.rectangle(img, (40,1), (140,65), (122,122,122), -1)
        img = cv2.rectangle(img, (160,1), (255,65), self.colors[0], -1)
        img = cv2.rectangle(img, (275,1), (370,65), self.colors[1], -1)
        img = cv2.rectangle(img, (390,1), (485,65), self.colors[2], -1)
        img = cv2.rectangle(img, (505,1), (600,65), self.colors[3], -1)
        cv2.putText(img, "CLEAR", (60, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        result = hands_detector.process(framergb)

        if result.multi_hand_landmarks:
            landmarks = []
            for handslms in result.multi_hand_landmarks:
                for lm in handslms.landmark:
                    landmarks.append([int(lm.x * img.shape[1]), int(lm.y * img.shape[0])])
                
                mp_draw.draw_landmarks(img, handslms, mp_hands.HAND_CONNECTIONS)
            
            # 
            center = (landmarks[8][0], landmarks[8][1])
            thumb = (landmarks[4][0], landmarks[4][1])

            # Check if thumb is near index finger (to "stop" drawing)
            if (thumb[1] - center[1] < 30):
                self.bpoints.append(deque(maxlen=512))
                self.blue_index += 1
                self.gpoints.append(deque(maxlen=512))
                self.green_index += 1
                self.rpoints.append(deque(maxlen=512))
                self.red_index += 1
                self.ypoints.append(deque(maxlen=512))
                self.yellow_index += 1

            elif center[1] <= 65:
                if 40 <= center[0] <= 140: # Clear
                    self.bpoints, self.gpoints, self.rpoints, self.ypoints = [deque(maxlen=512)], [deque(maxlen=512)], [deque(maxlen=512)], [deque(maxlen=512)]
                    self.blue_index = self.green_index = self.red_index = self.yellow_index = 0
                elif 160 <= center[0] <= 255: self.colorIndex = 0
                elif 275 <= center[0] <= 370: self.colorIndex = 1
                elif 390 <= center[0] <= 485: self.colorIndex = 2
                elif 505 <= center[0] <= 600: self.colorIndex = 3
            else:
                if self.colorIndex == 0: self.bpoints[self.blue_index].appendleft(center)
                elif self.colorIndex == 1: self.gpoints[self.green_index].appendleft(center)
                elif self.colorIndex == 2: self.rpoints[self.red_index].appendleft(center)
                elif self.colorIndex == 3: self.ypoints[self.yellow_index].appendleft(center)
        else:
            # Handle frame breaks when no hand is visible
            self.bpoints.append(deque(maxlen=512))
            self.blue_index += 1
            self.gpoints.append(deque(maxlen=512))
            self.green_index += 1
            self.rpoints.append(deque(maxlen=512))
            self.red_index += 1
            self.ypoints.append(deque(maxlen=512))
            self.yellow_index += 1

        # Render points
        points = [self.bpoints, self.gpoints, self.rpoints, self.ypoints]
        for i in range(len(points)):
            for j in range(len(points[i])):
                for k in range(1, len(points[i][j])):
                    if points[i][j][k - 1] is None or points[i][j][k] is None:
                        continue
                    cv2.line(img, points[i][j][k - 1], points[i][j][k], self.colors[i], 2)
        
        # Return frame in the format WebRTC expects
        import av
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- Run Streamlit ---
webrtc_streamer(
    key="whiteboard", 
    video_processor_factory=MyVideoProcessor,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}, # Add STUN server for better connectivity
    media_stream_constraints={"video": True, "audio": False},
)
