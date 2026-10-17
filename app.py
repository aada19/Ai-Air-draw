
import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import threading

from streamlit_webrtc import (
    webrtc_streamer,
    WebRtcMode,
    VideoProcessorBase,
)

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="AIR DRAW AI",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700;800&family=Poppins:wght@400;500;600&display=swap');

.stApp {
    background: linear-gradient(135deg, #080b18, #11152b, #10102a);
    color: #f1f5ff;
}

h1, h2, h3 {
    font-family: 'Orbitron', sans-serif !important;
}

.hero {
    text-align: center;
    padding: 22px 10px;
    border: 1px solid #00eaff55;
    border-radius: 18px;
    background: linear-gradient(120deg, #111a35, #21133c);
    box-shadow: 0 0 28px #00eaff12;
    margin-bottom: 22px;
}

.hero h1 {
    font-size: clamp(28px, 5vw, 48px);
    background: linear-gradient(90deg, #00eaff, #bd82ff, #ff61bd);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero p {
    color: #b4c4e4;
    font-family: 'Poppins', sans-serif;
}

div[data-testid="stMetric"] {
    background: #151b32;
    padding: 16px;
    border: 1px solid #3c4c75;
    border-radius: 12px;
}

div[data-testid="stMetricLabel"] {
    color: #8beeff;
}

.stButton button {
    border-radius: 10px;
    font-weight: 600;
}

.tip {
    background: #121a30;
    border-left: 4px solid #00eaff;
    padding: 13px 16px;
    border-radius: 8px;
    color: #d4e2ff;
    margin: 12px 0;
}
</style>

<div class="hero">
    <h1>✦ AIR DRAW AI ✦</h1>
    <p>DRAW IN THE AIR. CREATE WITHOUT TOUCHING.</p>
    <p>HAND TRACKING • COMPUTER VISION • REAL-TIME CREATIVITY</p>
</div>
""", unsafe_allow_html=True)


# --------------------------------------------------
# DRAWING SETTINGS
# --------------------------------------------------

WIDTH = 640
HEIGHT = 480

BACKGROUND = (17, 21, 40)  # BGR format

COLORS = {
    "Neon Cyan": (255, 235, 0),
    "Electric Pink": (210, 70, 255),
    "Lime Green": (70, 255, 100),
    "Purple": (255, 100, 180),
    "White": (245, 245, 245),
    "Orange": (0, 150, 255),
}


def create_canvas():
    canvas = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    canvas[:] = BACKGROUND

    # Decorative digital grid
    for x in range(0, WIDTH, 40):
        cv2.line(canvas, (x, 0), (x, HEIGHT), (27, 34, 57), 1)

    for y in range(0, HEIGHT, 40):
        cv2.line(canvas, (0, y), (WIDTH, y), (27, 34, 57), 1)

    cv2.putText(
        canvas,
        "YOUR DIGITAL CANVAS",
        (18, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (130, 155, 195),
        1,
        cv2.LINE_AA,
    )

    return canvas


# --------------------------------------------------
# SHARED STATE
# --------------------------------------------------

class DrawingState:
    def __init__(self):
        self.lock = threading.Lock()
        self.canvas = create_canvas()
        self.color = COLORS["Neon Cyan"]
        self.brush_size = 7
        self.eraser = False
        self.hits = 0
        self.draw_active = False

    def clear(self):
        with self.lock:
            self.canvas = create_canvas()
            self.hits = 0


state = DrawingState()


# --------------------------------------------------
# MEDIAPIPE HAND TRACKING
# --------------------------------------------------

class AirDrawProcessor(VideoProcessorBase):
    def __init__(self, drawing_state):
        self.state = drawing_state
        self.previous_point = None

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=0,
            min_detection_confidence=0.65,
            min_tracking_confidence=0.60,
        )

    def recv(self, frame):
        # Get webcam frame
        image = frame.to_ndarray(format="bgr24")

        # Mirror image like a selfie camera
        image = cv2.flip(image, 1)
        image = cv2.resize(image, (WIDTH, HEIGHT))

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        drawing_now = False
        finger_point = None
        message = "SHOW YOUR HAND"

        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]
            landmarks = hand.landmark

            # Index and middle finger landmarks
            index_tip = landmarks[8]
            index_pip = landmarks[6]
            middle_tip = landmarks[12]
            middle_pip = landmarks[10]

            # Draw when index is up and middle finger is down
            index_up = index_tip.y < index_pip.y
            middle_up = middle_tip.y < middle_pip.y

            x = int(index_tip.x * WIDTH)
            y = int(index_tip.y * HEIGHT)

            finger_point = (
                max(0, min(WIDTH - 1, x)),
                max(0, min(HEIGHT - 1, y)),
            )

            if index_up and not middle_up:
                drawing_now = True
                message = "DRAWING MODE"
            elif index_up and middle_up:
                message = "PAUSE / MOVE"
            else:
                message = "RAISE INDEX FINGER"

            # Draw on shared canvas
            with self.state.lock:
                if drawing_now:
                    color = (
                        BACKGROUND
                        if self.state.eraser
                        else self.state.color
                    )

                    size = (
                        max(12, self.state.brush_size * 2)
                        if self.state.eraser
                        else self.state.brush_size
                    )

                    if self.previous_point is not None:
                        cv2.line(
                            self.state.canvas,
                            self.previous_point,
                            finger_point,
                            color,
                            size,
                            cv2.LINE_AA,
                        )

                    cv2.circle(
                        self.state.canvas,
                        finger_point,
                        max(3, size // 2),
                        color,
                        -1,
                        cv2.LINE_AA,
                    )

                    self.state.hits += 1

            if drawing_now:
                self.previous_point = finger_point
            else:
                self.previous_point = None

            # Draw hand landmarks on webcam preview
            mp.solutions.drawing_utils.draw_landmarks(
                image,
                hand,
                self.mp_hands.HAND_CONNECTIONS,
            )

            cv2.circle(
                image,
                finger_point,
                10,
                (255, 255, 255),
                2,
            )

        else:
            self.previous_point = None

        # Camera preview labels
        cv2.rectangle(image, (0, 0), (WIDTH, 48), (15, 20, 40), -1)

        cv2.putText(
            image,
            "AIR DRAW AI | LIVE CAMERA",
            (15, 29),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 235, 0),
            2,
            cv2.LINE_AA,
        )

        status_color = (
            (80, 255, 120)
            if drawing_now
            else (120, 190, 255)
        )

        cv2.putText(
            image,
            message,
            (15, HEIGHT - 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            status_color,
            2,
            cv2.LINE_AA,
        )

        # Copy canvas safely for live display
        with self.state.lock:
            canvas = self.state.canvas.copy()

        # Display webcam and canvas side by side
        combined = np.hstack((image, canvas))

        return av_frame(combined)


# --------------------------------------------------
# VIDEO FRAME CONVERSION
# --------------------------------------------------

def av_frame(image):
    import av
    return av.VideoFrame.from_ndarray(image, format="bgr24")


# --------------------------------------------------
# SIDEBAR CONTROLS
# --------------------------------------------------

st.sidebar.title("🎨 DRAWING STUDIO")
st.sidebar.caption("Customise your digital brush.")

selected_color = st.sidebar.selectbox(
    "Choose brush colour",
    list(COLORS.keys()),
)

brush_size = st.sidebar.slider(
    "Brush thickness",
    min_value=2,
    max_value=20,
    value=7,
)

eraser = st.sidebar.toggle(
    "Eraser mode",
    value=False,
)

with state.lock:
    state.color = COLORS[selected_color]
    state.brush_size = brush_size
    state.eraser = eraser

st.sidebar.markdown("---")
st.sidebar.markdown("### ✨ Gesture Guide")
st.sidebar.markdown("""
- ☝️ **Index finger up:** Draw
- ✌️ **Index + middle up:** Pause drawing
- ✊ **Lower index finger:** Stop drawing
""")

st.sidebar.markdown("---")
st.sidebar.info(
    "Allow camera access when your browser asks for permission."
)


# --------------------------------------------------
# MAIN DASHBOARD
# --------------------------------------------------

st.markdown("### 🖥️ Live Air Drawing Studio")

left, right = st.columns([2, 1])

with left:
    st.markdown("#### 🎥 Camera + Digital Canvas")
    st.caption(
        "The camera appears on the left and your drawing on the right."
    )

with right:
    st.markdown("#### 🕹️ Canvas Controls")

    if st.button("🗑️ Clear Drawing", use_container_width=True):
        state.clear()

    if st.button("↩️ Reset Canvas", use_container_width=True):
        state.clear()

st.markdown(
    '<div class="tip">💡 Tip: Keep your hand visible, '
    'raise only your index finger to draw, and raise two fingers '
    'to move without drawing.</div>',
    unsafe_allow_html=True,
)

# WebRTC must run through the browser's camera.
webrtc_ctx = webrtc_streamer(
    key="air-draw-camera",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=lambda: AirDrawProcessor(state),
    media_stream_constraints={
        "video": True,
        "audio": False,
    },
    async_processing=True,
)

st.markdown("---")
st.markdown("### 🧠 How the AI Vision System Works")

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("#### 01 · DETECT")
    st.write("The webcam captures your hand in real time.")

with c2:
    st.markdown("#### 02 · TRACK")
    st.write("MediaPipe tracks your hand landmarks and index fingertip.")

with c3:
    st.markdown("#### 03 · CREATE")
    st.write("The program maps fingertip movement into digital brush strokes.")

st.caption(
    "AIR DRAW AI | Computer Vision • Gesture Recognition • Creative Coding"
)
