import sys
import os
import threading
import time
import psutil  # For system monitoring
import random  # For simulation (remove when linking real model)

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QTextEdit, QStackedWidget, 
                             QLabel, QFrame, QSlider, QLineEdit, QGridLayout,
                             QGraphicsDropShadowEffect)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QObject, QTimer, QSize
from PyQt5.QtGui import QColor, QFont, QIcon

# --- Advanced Graphing Library ---
import pyqtgraph as pg

# --- FFPyPlayer for Audio/Video ---
from ffpyplayer.player import MediaPlayer
from PyQt5.QtGui import QImage, QPixmap

# --- Import your actual model ---
# Assuming 'model.py' is in the same directory
try:
    import model
except ImportError:
    print("Warning: 'model.py' not found. Running in simulation mode.")
    model = None

# ==========================================
# 1. UI COMPONENTS & STYLING
# ==========================================
STYLESHEET = """
    QMainWindow { background-color: #0b0c15; }
    
    /* Panels */
    QFrame#Panel { 
        background-color: #151725; 
        border: 1px solid #2a2d3e; 
        border-radius: 10px; 
    }
    
    /* Labels */
    QLabel { color: #aab2c0; font-family: 'Segoe UI', sans-serif; }
    QLabel#Title { color: white; font-size: 18px; font-weight: bold; }
    QLabel#StatValue { color: #00e5ff; font-size: 22px; font-weight: bold; }
    QLabel#StatLabel { color: #6c7293; font-size: 12px; }

    /* Buttons */
    QPushButton {
        background-color: #1f2336;
        color: white;
        border: 1px solid #00e5ff;
        padding: 12px;
        border-radius: 6px;
        font-weight: bold;
        font-size: 14px;
    }
    QPushButton:hover { background-color: #00e5ff; color: black; }
    QPushButton:pressed { background-color: #00b8cc; }
    QPushButton:disabled { border: 1px solid #333; color: #555; background-color: #111; }

    /* Inputs */
    QLineEdit {
        background-color: #0b0c15; color: #00e5ff; 
        border: 1px solid #2a2d3e; padding: 8px; border-radius: 4px;
    }
    
    /* Logs */
    QTextEdit {
        background-color: #08090f; color: #00ff9d; 
        font-family: 'Consolas', monospace; font-size: 12px;
        border: 1px solid #2a2d3e; border-radius: 6px;
    }
"""

class ModernFrame(QFrame):
    def __init__(self):
        super().__init__()
        self.setObjectName("Panel")
        # Add shadow for depth
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(15)
        shadow.setColor(QColor(0, 0, 0, 150))
        shadow.setOffset(0, 4)
        self.setGraphicsEffect(shadow)

# ==========================================
# 2. VIDEO PLAYER (Optimized)
# ==========================================
class FFVideoWidget(QLabel):
    video_finished = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background-color: black;")
        self.player = None
        self.timer = QTimer()
        self.timer.setInterval(20) # 50 FPS target
        self.timer.timeout.connect(self.update_frame)

    def play(self, path):
        self.stop()
        if not os.path.exists(path):
            print(f"Video not found: {path}")
            self.video_finished.emit()
            return
            
        ff_opts = {'out_fmt': 'rgb24'}
        self.player = MediaPlayer(path, ff_opts=ff_opts)
        self.timer.start()

    def update_frame(self):
        if self.player is None: return
        frame, val = self.player.get_frame()
        if val == 'eof':
            self.stop()
            self.video_finished.emit()
            return
        if frame is not None:
            img, t = frame
            w, h = img.get_size()
            data = img.to_bytearray()[0]
            qimg = QImage(data, w, h, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg).scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.setPixmap(pixmap)

    def stop(self):
        self.timer.stop()
        if self.player:
            self.player.close_player()
            self.player = None
        self.clear()

# ==========================================
# 3. WORKER THREADS (Logic)
# ==========================================
class TrainingWorker(QThread):
    log_signal = pyqtSignal(str)
    data_signal = pyqtSignal(float, float) # epoch, loss
    finished_signal = pyqtSignal()
    
    def __init__(self, epochs, lr):
        super().__init__()
        self.epochs = epochs
        self.lr = lr
        self._is_running = True

    def run(self):
        self.log_signal.emit(f">>> INITIALIZING CORE. Epochs: {self.epochs} | LR: {self.lr}")
        
        # --- IF USING REAL MODEL ---
        # If your model.py has a function like train(callback), pass a callback here.
        # For now, we simulate the training loop to show off the Graphing GUI.
        
        current_loss = 1.0
        try:
            for i in range(self.epochs):
                if not self._is_running: break
                
                # Simulate processing time
                time.sleep(0.1) 
                
                # Math simulation for the graph
                current_loss = current_loss * 0.95 + (random.random() * 0.05)
                accuracy = 1.0 - current_loss
                
                # Emit data to GUI
                self.log_signal.emit(f"Epoch {i+1}/{self.epochs} - Loss: {current_loss:.4f}")
                self.data_signal.emit(i+1, current_loss)
                
            # Call actual model if needed:
            # if model: model.main() 
            
        except Exception as e:
            self.log_signal.emit(f"CRITICAL ERROR: {str(e)}")
            
        self.finished_signal.emit()

    def stop(self):
        self._is_running = False

# ==========================================
# 4. MAIN DASHBOARD
# ==========================================
class Dashboard(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Cognara AI // Predictive Core")
        self.resize(1200, 800)
        self.setStyleSheet(STYLESHEET)

        # Main Stack (Video vs App)
        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)

        # 1. Video Layer
        self.video_player = FFVideoWidget()
        self.video_player.video_finished.connect(self.on_intro_finished)
        self.stack.addWidget(self.video_player)

        # 2. Control Panel Layer
        self.control_widget = QWidget()
        self.setup_ui()
        self.stack.addWidget(self.control_widget)

        # Initialize
        self.is_training = False
        
        # System Monitor Timer
        self.sys_timer = QTimer()
        self.sys_timer.timeout.connect(self.update_system_stats)
        self.sys_timer.start(1000)

        # Start Intro
        self.video_player.play("HALO.mp4")

    def setup_ui(self):
        main_layout = QHBoxLayout(self.control_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)

        # --- LEFT SIDEBAR (Controls) ---
        sidebar = QVBoxLayout()
        
        # Title Card
        title_card = ModernFrame()
        tc_layout = QVBoxLayout(title_card)
        t_label = QLabel("NEURAL\nINTERFACE")
        t_label.setObjectName("Title")
        t_label.setAlignment(Qt.AlignCenter)
        tc_layout.addWidget(t_label)
        sidebar.addWidget(title_card)

        # Parameters Card
        param_card = ModernFrame()
        pc_layout = QVBoxLayout(param_card)
        pc_layout.addWidget(QLabel("Epochs"))
        self.epoch_input = QLineEdit("100")
        pc_layout.addWidget(self.epoch_input)
        
        pc_layout.addWidget(QLabel("Learning Rate"))
        self.lr_input = QLineEdit("0.001")
        pc_layout.addWidget(self.lr_input)
        
        pc_layout.addStretch()
        sidebar.addWidget(param_card, 1) # Stretch factor 1

        # Actions
        self.btn_train = QPushButton("INITIATE SEQUENCE")
        self.btn_train.clicked.connect(self.toggle_training)
        self.btn_stop = QPushButton("ABORT")
        self.btn_stop.clicked.connect(self.abort_training)
        self.btn_stop.setEnabled(False)
        self.btn_stop.setStyleSheet("border: 1px solid #ff0055; color: #ff0055;")
        
        sidebar.addWidget(self.btn_train)
        sidebar.addWidget(self.btn_stop)

        # --- CENTER (Graphs & Stats) ---
        center_col = QVBoxLayout()

        # Stats Row
        stats_layout = QHBoxLayout()
        self.cpu_stat = self.create_stat_widget("CPU LOAD", "0%")
        self.ram_stat = self.create_stat_widget("MEMORY", "0 GB")
        self.acc_stat = self.create_stat_widget("EST. ACCURACY", "0.00%")
        stats_layout.addWidget(self.cpu_stat)
        stats_layout.addWidget(self.ram_stat)
        stats_layout.addWidget(self.acc_stat)
        center_col.addLayout(stats_layout)

        # Graph Area
        graph_card = ModernFrame()
        gc_layout = QVBoxLayout(graph_card)
        
        # Setup PyQtGraph
        pg.setConfigOption('background', '#0f111a')
        pg.setConfigOption('foreground', '#aab2c0')
        self.plot_widget = pg.PlotWidget(title="Loss Convergence")
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_curve = self.plot_widget.plot(pen=pg.mkPen(color='#00e5ff', width=2))
        
        gc_layout.addWidget(self.plot_widget)
        center_col.addWidget(graph_card, 2)

        # Terminal
        term_card = ModernFrame()
        term_layout = QVBoxLayout(term_card)
        term_layout.addWidget(QLabel("System Log"))
        self.terminal = QTextEdit()
        self.terminal.setReadOnly(True)
        term_layout.addWidget(self.terminal)
        center_col.addWidget(term_card, 1)

        # Add Layouts
        main_layout.addLayout(sidebar, 1)
        main_layout.addLayout(center_col, 3)

    def create_stat_widget(self, title, initial_val):
        frame = ModernFrame()
        layout = QVBoxLayout(frame)
        val_lbl = QLabel(initial_val)
        val_lbl.setObjectName("StatValue")
        lbl = QLabel(title)
        lbl.setObjectName("StatLabel")
        layout.addWidget(val_lbl)
        layout.addWidget(lbl)
        layout.setAlignment(Qt.AlignCenter)
        frame.val_label = val_lbl # Store reference
        return frame

    # --- LOGIC ---
    def on_intro_finished(self):
        self.video_player.stop()
        self.stack.setCurrentIndex(1) # Show Dashboard

    def update_system_stats(self):
        cpu = psutil.cpu_percent()
        ram = psutil.virtual_memory().percent
        self.cpu_stat.val_label.setText(f"{cpu}%")
        self.ram_stat.val_label.setText(f"{ram}%")

    def log(self, text):
        self.terminal.append(text)
        # Auto scroll
        sb = self.terminal.verticalScrollBar()
        sb.setValue(sb.maximum())

    def toggle_training(self):
        if self.is_training: return

        # Validate Inputs
        try:
            epochs = int(self.epoch_input.text())
            lr = float(self.lr_input.text())
        except ValueError:
            self.log(">>> ERROR: Invalid Parameters.")
            return

        self.is_training = True
        self.btn_train.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.epoch_data = []
        self.loss_data = []
        
        self.log(">>> ESTABLISHING NEURAL LINK...")
        
        # Start Worker
        self.worker = TrainingWorker(epochs, lr)
        self.worker.log_signal.connect(self.log)
        self.worker.data_signal.connect(self.update_graph)
        self.worker.finished_signal.connect(self.on_training_complete)
        self.worker.start()

    def update_graph(self, epoch, loss):
        self.epoch_data.append(epoch)
        self.loss_data.append(loss)
        self.plot_curve.setData(self.epoch_data, self.loss_data)
        
        # Update Accuracy Stat (Inverse of loss for visual flair)
        acc = max(0, (1.0 - loss) * 100)
        self.acc_stat.val_label.setText(f"{acc:.2f}%")

    def abort_training(self):
        if self.worker:
            self.worker.stop()
            self.log(">>> INTERRUPT SIGNAL SENT.")

    def on_training_complete(self):
        self.is_training = False
        self.btn_train.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_train.setText("RE-INITIALIZE")
        self.log(">>> SEQUENCE COMPLETE.")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Dashboard()
    window.show()
    sys.exit(app.exec_())