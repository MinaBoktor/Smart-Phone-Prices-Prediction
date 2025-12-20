import sys
import os
import threading
import time
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QPushButton, QTextEdit, QStackedWidget, QLabel)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QObject, QTimer
from PyQt5.QtGui import QImage, QPixmap

# --- FFPyPlayer for Synced Audio/Video ---
from ffpyplayer.player import MediaPlayer

import model 

# --- 1. Output Redirector ---
class StreamRedirector(QObject):
    text_written = pyqtSignal(str)
    
    def write(self, text):
        self.text_written.emit(str(text))
        
    def flush(self):
        pass

# --- 2. Worker Thread ---
class TrainingWorker(QThread):
    finished_signal = pyqtSignal()
    
    def run(self):
        try:
            print(">>> SYSTEM: Starting Model Training...", flush=True)
            print(">>> NOTE: This process may take several minutes.", flush=True)
            print(">>> NOTE: If text stops appearing, the model is crunching numbers. Please wait.", flush=True)
            model.main()
        except Exception as e:
            print(f"CRITICAL ERROR IN MODEL: {e}", flush=True)
        finally:
            self.finished_signal.emit()

# --- 3. Robust Video Widget ---
class FFVideoWidget(QLabel):
    video_finished = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background-color: black;")
        self.player = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

    def play(self, path):
        if self.player:
            self.player.close_player()
            self.player = None
        
        # 'out_fmt': 'rgb24' ensures compatibility with PyQt
        ff_opts = {'out_fmt': 'rgb24'}
        try:
            self.player = MediaPlayer(path, ff_opts=ff_opts)
            # Give it a moment to initialize
            time.sleep(0.1)
            self.timer.start(10) # Start polling
        except Exception as e:
            print(f"Error loading video: {e}")
            self.video_finished.emit()

    def update_frame(self):
        if self.player is None:
            return

        # Get frame and delay (val)
        frame, val = self.player.get_frame()

        # 'eof' means End of File
        if val == 'eof':
            self.stop()
            self.video_finished.emit()
            return
        
        if frame is not None:
            img, t = frame
            # Convert to QImage
            w, h = img.get_size()
            data = img.to_bytearray()[0]
            qimg = QImage(data, w, h, QImage.Format_RGB888)
            
            # Scale to window
            pixmap = QPixmap.fromImage(qimg).scaled(
                self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.setPixmap(pixmap)
            
            # Adjust timer speed based on video requirement (val is in seconds)
            # We cap it to avoid UI freeze (min 5ms)
            delay_ms = max(5, int(val * 1000))
            self.timer.setInterval(delay_ms)

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
        
        # Stylesheet
        self.setStyleSheet("""
            QMainWindow { background-color: #0f0f0f; }
            QPushButton {
                background-color: #007acc; color: white; border: none;
                padding: 15px; font-size: 16px; font-weight: bold; border-radius: 5px;
            }
            QPushButton:hover { background-color: #005f9e; }
            QPushButton:disabled { background-color: #333; color: #777; }
            QTextEdit {
                background-color: #1e1e1e; color: #00ff00; font-family: Consolas, Monospace;
                font-size: 14px; border: 1px solid #333;
            }
            QLabel { color: white; font-size: 24px; font-weight: bold; }
        """)

        self.central_stack = QStackedWidget()
        self.setCentralWidget(self.central_stack)

        # Layer 0: Video
        self.video_widget = FFVideoWidget()
        self.video_widget.video_finished.connect(self.on_media_finished)
        self.central_stack.addWidget(self.video_widget)

        # Layer 1: App
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

        # State
        self.state = "INTRO"
        self.force_close = False

        # Redirector
        self.redirector = StreamRedirector()
        self.redirector.text_written.connect(self.update_terminal)
        sys.stdout = self.redirector

        # Start App
        self.play_video("HALO.mp4")

    def play_video(self, filename):
        path = os.path.abspath(filename)
        if os.path.exists(path):
            self.central_stack.setCurrentIndex(0)
            self.video_widget.play(path)
        else:
            print(f"Warning: File {filename} not found.")
            # Important: Don't auto-recurse here to prevent infinite loop
            if self.state == "INTRO": self.show_app()
            elif self.state == "EXIT": self.close_application()

    def on_media_finished(self):
        if self.state == "INTRO":
            self.show_app()
        elif self.state == "TRAINING":
            # Safety Check: Only loop if we are truly still training
            if self.worker.isRunning():
                path = os.path.abspath("FETCH.mp4")
                if os.path.exists(path):
                    self.video_widget.play(path)
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
        self.train_btn.setText("TRAINING IN PROGRESS (Please Wait...)")
        
        self.play_video("FETCH.mp4")
        
        self.worker = TrainingWorker()
        self.worker.finished_signal.connect(self.on_training_finished)
        self.worker.start()

    def on_training_finished(self):
        self.state = "APP"
        self.video_widget.stop()
        self.central_stack.setCurrentIndex(1)
        self.train_btn.setEnabled(True)
        self.train_btn.setText("INITIALIZE MODEL TRAINING")
        print("\n>>> SYSTEM: Training Sequence Complete.")

    def update_terminal(self, text):
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