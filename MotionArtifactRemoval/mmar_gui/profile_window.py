import numpy as np
from matplotlib.backends.backend_qt5agg import \
    FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import QSizePolicy


class BgProfileWindow(FigureCanvas):

    def __init__(self, width=6, height=4, dpi=100):

        self.fig = Figure(figsize=(width, height), dpi=dpi)
        FigureCanvas.__init__(self, self.fig)

        FigureCanvas.setSizePolicy(self,
                                   QSizePolicy.Expanding,
                                   QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        
    def updatefig(self, foreground, background, mouse=None, z_idx=None, frame_num=None):
        xx = np.arange(1, len(foreground)+1)
        self.fig.clear()
        self.axes = self.fig.add_subplot(111)
        self.axes.plot(xx, foreground, marker='o', color='dodgerblue', label='Foreground')
        if frame_num:
            self.axes.plot(frame_num, foreground[frame_num-1], marker='o', color='red')
        self.axes.title.set_text(mouse+' Slice #'+str(z_idx))
        self.axes.title.set_fontsize(10)
        self.axes.set_xlabel('Slice Number')
        self.axes.set_ylabel('Foreground Intensity', color='dodgerblue')
        self.axes2 = self.axes.twinx()

        self.axes2.plot(xx, background, marker='o', color='darkorange', label='Background')
        self.axes2.set_ylabel('Background Intensity', color='darkorange')
        self.show()
