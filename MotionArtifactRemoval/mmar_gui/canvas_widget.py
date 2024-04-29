
from PyQt5 import QtCore
from PyQt5.QtWidgets import QVBoxLayout, QWidget
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import numpy as np


class CanvasWidget(QWidget):

    canvasUpdated = QtCore.pyqtSignal(bool)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.figure = plt.figure(figsize=(3,2))
        self.canvas = FigureCanvas(self.figure)

        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)

    def clear(self):
        ''' Clear the display of the image '''
        self.figure.clear()
        self.canvas.draw()



    def update_image(self, image, contrast_range):
        """
        Updates the image on the canvas.
        """
        if image is None:
            return

        self.figure.clear()
        ax = self.figure.add_subplot(111)

        if contrast_range is None:
            ax.imshow(image, cmap='gray')
        else:
            max_val = np.max(image)
            new_min = max_val * contrast_range[0]
            new_max = max_val * contrast_range[1]
            ax.imshow(image, cmap='gray', vmin=new_min, vmax=new_max)
        plt.axis('off')
        self.figure.tight_layout()
        self.canvas.draw()
