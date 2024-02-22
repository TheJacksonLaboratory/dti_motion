
import matplotlib
matplotlib.use("Qt5Agg")

from PyQt5 import QtCore
from PyQt5.QtWidgets import QVBoxLayout, QWidget

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import \
    FigureCanvasQTAgg as FigureCanvas


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



    def update_image(self, image, contrast_range, display_mask=False):
        """
        Updates the image on the canvas.
        """
        if image is None:
            return
        self.figure.clear()
        ax = self.figure.add_subplot(111)

        if display_mask:
            ax.imshow(image, cmap='gray')
        else:
            # for frames (except the frame 0) we keep the range the same so imshow doesn't adjust contrast automatically
            if contrast_range is None:
                ax.imshow(image, cmap='gray')
            else:
                ax.imshow(
                        image, 
                        cmap='gray', 
                        vmin=contrast_range[0],
                        vmax=contrast_range[1]
                    )
        plt.axis('off')
        self.figure.tight_layout()
        self.canvas.draw()
