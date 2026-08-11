"""Launch the GUI for 4 seconds on the real platform, then auto-quit."""
import sys
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication
from forge_gui.app import MainWindow

app = QApplication(sys.argv)
w = MainWindow()
w.show()
QTimer.singleShot(4000, app.quit)
print("launched real platform — auto-quit in 4s")
sys.exit(app.exec())
