import sys
import os
import threading
import time
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QPushButton, QTextEdit, QStackedWidget, QLabel)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QObject, QTimer
from PyQt5.QtGui import QImage, QPixmap

# --- FFPyPlayer for Audio/Video ---
from ffpyplayer.player import MediaPlayer

import model 

# --- 1. Buffered Output Redirector (Fixes Text Lag) ---
class StreamRedirector(QObject):
    # We no longer emit signal on every write. We buffer it.
    def __init__(self):
        super().__init__()
        self.buffer = ""
        self.lock = threading.Lock()

    def write(self, text):
        with self.lock:
            self.buffer += str(text)

    def flush(self):
        pass
        
    def read_and_clear(self):
        with self.lock:
            data = self.buffer
            self.buffer = ""
        return data

# --- 2. Worker Thread ---
class TrainingWorker(QThread):
    finished_signal = pyqtSignal()
    
    def run(self):
        try:
            print(">>> SYSTEM: Core Protocols Engaged...", flush=True)
            print(">>> ALLOCATING RESOURCES: Please wait...", flush=True)
            # Short sleep to let video start smoothly before heavy load
            time.sleep(1) 
            model.main()
        except Exception as e:
            print(f"CRITICAL ERROR: {e}", flush=True)
        finally:
            self.finished_signal.emit()

# --- 3. High-Performance Video Widget ---
class FFVideoWidget(QLabel):
    video_finished = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background-color: black;")
        self.player = None
        
        # We poll faster (30ms) to catch frames immediately
        self.timer = QTimer()
        self.timer.setInterval(30) 
        self.timer.timeout.connect(self.update_frame)

    def play(self, path):
        self.stop()
        
        # 'out_fmt': 'rgb24' is fastest for Qt conversion
        ff_opts = {'out_fmt': 'rgb24'}
        try:
            self.player = MediaPlayer(path, ff_opts=ff_opts)
            self.timer.start()
        except Exception as e:
            print(f"Video Error: {e}")
            self.video_finished.emit()

    def update_frame(self):
        if self.player is None:
            return

        # Grab frame
        frame, val = self.player.get_frame()

        if val == 'eof':
            self.stop()
            self.video_finished.emit()
            return
        
        if frame is not None:
            img, t = frame
            w, h = img.get_size()
            
            # Fast raw data access
            data = img.to_bytearray()[0]
            
            # Create QImage
            qimg = QImage(data, w, h, QImage.Format_RGB888)
            
            # Scale and Set
            pixmap = QPixmap.fromImage(qimg).scaled(
                self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.setPixmap(pixmap)

    def stop(self):
        self.timer.stop()
        if self.player:
            self.player.close_player()
            self.player = None
        self.clear()

# --- 4. Main Application ---
class AdvancedApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Smart Phone Price Predictor - AI Core")
        self.resize(1000, 700)
        
        self.setStyleSheet("""
            QMainWindow { background-color: #0f0f0f; }
            QPushButton {
                background-color: #007acc; color: white; border: none;
                padding: 15px; font-size: 16px; font-weight: bold; border-radius: 5px;
            }
            QPushButton:hover { background-color: #005f9e; }
            QPushButton:disabled { background-color: #333; color: #555; }
            QTextEdit {
                background-color: #1e1e1e; color: #00ff00; font-family: Consolas, Monospace;
                font-size: 14px; border: 1px solid #333;
            }
            QLabel { color: white; font-size: 24px; font-weight: bold; }
        """)

        self.central_stack = QStackedWidget()
        self.setCentralWidget(self.central_stack)

        # Layers
        self.video_widget = FFVideoWidget()
        self.video_widget.video_finished.connect(self.on_media_finished)
        self.central_stack.addWidget(self.video_widget)

        self.app_container = QWidget()
        self.app_layout = QVBoxLayout(self.app_container)
        
        self.header_label = QLabel("NEURAL NETWORK COMMAND CENTER")
        self.header_label.setAlignment(Qt.AlignCenter)
        self.app_layout.addWidget(self.header_label)

        self.terminal_display = QTextEdit()
        self.terminal_display.setReadOnly(True)
        self.app_layout.addWidget(self.terminal_display)

        self.train_btn = QPushButton("INITIALIZE MODEL TRAINING")
        self.train_btn.clicked.connect(self.start_training)
        self.app_layout.addWidget(self.train_btn)

        self.central_stack.addWidget(self.app_container)

        self.state = "INTRO"
        self.force_close = False

        # --- SETUP BUFFERED LOGGING ---
        self.redirector = StreamRedirector()
        sys.stdout = self.redirector
        
        # GUI Timer to flush text buffer every 100ms
        self.log_timer = QTimer()
        self.log_timer.timeout.connect(self.process_logs)
        self.log_timer.start(100)

        # START
        self.play_video("HALO.mp4")

    def play_video(self, filename):
        path = os.path.abspath(filename)
        if os.path.exists(path):
            self.central_stack.setCurrentIndex(0)
            self.video_widget.play(path)
        else:
            print(f"Warning: {filename} missing.")
            self.on_media_finished()

    def on_media_finished(self):
        if self.state == "INTRO":
            self.show_app()
        elif self.state == "TRAINING":
            # Loop loading video if worker is still running
            if self.worker.isRunning():
                self.play_video("FETCH.mp4")
            else:
                self.on_training_finished()
        elif self.state == "EXIT":
            self.close_application()

    def show_app(self):
        self.state = "APP"
        self.video_widget.stop()
        self.central_stack.setCurrentIndex(1)

    def start_training(self):
        self.state = "TRAINING"
        self.train_btn.setEnabled(False)
        self.train_btn.setText("TRAINING IN PROGRESS...")
        
        self.play_video("FETCH.mp4")
        
        self.worker = TrainingWorker()
        self.worker.finished_signal.connect(self.on_training_finished)
        self.worker.start()

    def on_training_finished(self):
        # Only stop if we aren't already stopping
        if self.state == "TRAINING":
            self.state = "APP"
            self.video_widget.stop()
            self.central_stack.setCurrentIndex(1)
            self.train_btn.setEnabled(True)
            self.train_btn.setText("INITIALIZE MODEL TRAINING")
            print("\n>>> SYSTEM: Training Sequence Complete.")

    def process_logs(self):
        """Reads buffer and updates GUI in one go"""
        text = self.redirector.read_and_clear()
        if text:
            cursor = self.terminal_display.textCursor()
            cursor.movePosition(cursor.End)
            cursor.insertText(text)
            self.terminal_display.setTextCursor(cursor)
            self.terminal_display.ensureCursorVisible()

    def closeEvent(self, event):
        if self.force_close:
            event.accept()
        else:
            event.ignore()
            self.state = "EXIT"
            self.play_video("BYE.mp4")

    def close_application(self):
        self.force_close = True
        self.video_widget.stop()
        self.close()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AdvancedApp()
    window.show()
    sys.exit(app.exec_())