import sys
import cv2
import numpy as np
import mediapipe as mp
from scipy.signal import butter, filtfilt, find_peaks

from PySide6.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QHBoxLayout, QWidget
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QImage, QPixmap

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# ================== Filter Sinyal ==================
def bandpass_filter(data, lowcut=0.7, highcut=4.0, fs=30, order=3):
    nyq = 0.5*fs
    low, high = lowcut/nyq, highcut/nyq
    b, a = butter(order, [low, high], btype='band')
    padlen = 3 * max(len(a), len(b))
    if len(data) <= padlen:
        return data  # jangan filter, kembalikan raw
    return filtfilt(b, a, data)

# ================== Skin Extractor ==================
def extract_skin(img):
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    mask = cv2.inRange(ycrcb, (0,135,85), (255,180,135))
    skin = cv2.bitwise_and(img, img, mask=mask)
    return skin, mask

# ================== Canvas Raw Signal ==================
class RawSignalCanvas(FigureCanvas):
    def __init__(self, max_points=300):
        self.fig = Figure(figsize=(6,3))
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.max_points = max_points
        self.signal = []
        self.line, = self.ax.plot([], [], color='green')
        self.ax.set_title("Raw Green Signal")
        self.ax.set_xlabel("Frame")
        self.ax.set_ylabel("Green Value")
   

    def update_signal(self, buffer):
        self.signal = list(buffer[-self.max_points:])
        frames = list(range(len(self.signal)))
        self.line.set_data(frames, self.signal)
        if self.signal:
            self.ax.set_xlim(0, len(self.signal))
            self.ax.set_ylim(min(self.signal)-5, max(self.signal)+5)
        self.draw()

# ================== Canvas Filtered Signal ==================
class FilteredSignalCanvas(FigureCanvas):
    def __init__(self, max_points=300):
        self.fig = Figure(figsize=(6,3))
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.max_points = max_points
        self.signal = []
        self.line, = self.ax.plot([], [], color='cyan')
        self.ax.set_title("Filtered rPPG Signal")
        self.ax.set_xlabel("Frame")
        self.ax.set_ylabel("Filtered Value")

    def update_signal(self, buffer):
        self.signal = list(buffer[-self.max_points:])
        frames = list(range(len(self.signal)))
        self.line.set_data(frames, self.signal)
        if self.signal:
            self.ax.set_xlim(0, len(self.signal))
            self.ax.set_ylim(min(self.signal)-5, max(self.signal)+5)
        self.draw()

# ================== Canvas BPM ==================
class BPMCanvas(FigureCanvas):
    def __init__(self, max_points=200):
        self.fig = Figure(figsize=(6,3))
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.max_points = max_points
        self.time_values = []
        self.bpm_values = []
        self.line, = self.ax.plot([], [], color='red')
        self.ax.set_title("Heart Rate (BPM)")
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("BPM")

    def update_bpm(self, bpm):
        t = self.time_values[-1]+1 if self.time_values else 0
        self.time_values.append(t)
        self.bpm_values.append(bpm)
        if len(self.time_values) > self.max_points:
            self.time_values = self.time_values[-self.max_points:]
            self.bpm_values = self.bpm_values[-self.max_points:]
        self.line.set_data(self.time_values, self.bpm_values)
        if self.time_values:
            self.ax.set_xlim(min(self.time_values), max(self.time_values))
        if self.bpm_values:
            self.ax.set_ylim(min(self.bpm_values)-5, max(self.bpm_values)+5)
        self.draw()

# ================== Main App ==================
class HRApp(QMainWindow):
    def __init__(self, video_path="vid.mp4"):
        super().__init__()
        self.setWindowTitle("rPPG Heart Rate Estimation")
        self.resize(600, 1500) 
        self.cap = cv2.VideoCapture(video_path)
        self.fs = self.cap.get(cv2.CAP_PROP_FPS) or 30
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5
        )

        self.buffer_size = 300
        self.signal_buffer = []

        # Widgets
        self.video_label = QLabel(alignment=Qt.AlignCenter)
        self.video_label.setFixedSize(640,480)
        self.face_label = QLabel(alignment=Qt.AlignCenter)
        self.face_label.setFixedSize(320,240)
        self.skin_label = QLabel(alignment=Qt.AlignCenter)
        self.skin_label.setFixedSize(320,240)
        self.bpm_label = QLabel("BPM: --", alignment=Qt.AlignCenter)
        self.bpm_label.setStyleSheet("font-size: 24px; color: red;")

        self.raw_canvas = RawSignalCanvas()
        self.filtered_canvas = FilteredSignalCanvas()
        self.bpm_canvas = BPMCanvas()

        # Layout
        h_layout = QHBoxLayout()
        h_layout.addWidget(self.face_label)
        h_layout.addWidget(self.skin_label)

        v_layout = QVBoxLayout()
        v_layout.addWidget(self.video_label)
        v_layout.addLayout(h_layout)
        v_layout.addWidget(self.raw_canvas)
        v_layout.addWidget(self.filtered_canvas)
        v_layout.addWidget(self.bpm_label)
        v_layout.addWidget(self.bpm_canvas)

        container = QWidget()
        container.setLayout(v_layout)
        self.setCentralWidget(container)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(int(1000/self.fs))

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES,0)
            return

        h, w, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)

        face_crop = np.zeros((200,200,3), dtype=np.uint8)
        skin_masked = np.zeros((200,200,3), dtype=np.uint8)
        green_mean = 0

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
            xs = [int(pt.x*w) for pt in landmarks.landmark]
            ys = [int(pt.y*h) for pt in landmarks.landmark]
            x_min, x_max = max(0,min(xs)), min(w,max(xs))
            y_min, y_max = max(0,min(ys)), min(h,max(ys))

            face_crop = frame[y_min:y_max, x_min:x_max]
            if face_crop.size>0:
                skin_masked, mask = extract_skin(face_crop)
                green_pixels = skin_masked[:,:,1][mask>0]
                if green_pixels.size>0:
                    green_mean = np.mean(green_pixels)
                self.face_label.setPixmap(self.to_qpixmap(face_crop))
                self.skin_label.setPixmap(self.to_qpixmap(skin_masked))

        # Update buffer
        self.signal_buffer.append(green_mean)
        if len(self.signal_buffer) > self.buffer_size:
            self.signal_buffer.pop(0)

        # Filtered signal
        filtered_signal = np.array(self.signal_buffer)
        if len(filtered_signal) > 10:  # minimal sample untuk filter
            filtered_signal = bandpass_filter(filtered_signal, fs=self.fs)

        # Update canvas
        self.raw_canvas.update_signal(self.signal_buffer)
        self.filtered_canvas.update_signal(filtered_signal)

        # Hitung BPM
        if len(self.signal_buffer) >= self.buffer_size:
            peaks, _ = find_peaks(filtered_signal, distance=int(self.fs*0.4))
            if len(peaks) > 1:
                intervals = np.diff(peaks)/self.fs
                bpm = 60/np.mean(intervals)
                self.bpm_label.setText(f"BPM: {int(bpm)}")
                self.bpm_canvas.update_bpm(bpm)

        self.video_label.setPixmap(self.to_qpixmap(frame))

    def to_qpixmap(self, img):
        if img.size==0: return QPixmap()
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h,w,ch = img_rgb.shape
        bytes_per_line = ch*w
        return QPixmap.fromImage(QImage(img_rgb.data, w,h,bytes_per_line, QImage.Format_RGB888))

    def closeEvent(self, event):
        self.cap.release()
        self.face_mesh.close()
        event.accept()

# ================== Main ==================
if __name__=="__main__":
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()  # ambil instance yang ada
    if app is None:
        app = QApplication(sys.argv)
 
    win = HRApp("vid.mp4")  # ganti path video
    win.show()
    sys.exit(app.exec())
