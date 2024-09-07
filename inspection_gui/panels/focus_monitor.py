import open3d.visualization.gui as gui
import matplotlib.pyplot as plt


class FocusMonitorPanel(gui.CollapsableVert):

    def __init__(self, s, em):
        super().__init__("Focus Monitor", 0, gui.Margins(
            s * em, s * em, s * em, s * em))

        self.metric_selector = gui.Combobox()
        self.figure = plt.figure()
        self.plot = gui.ImageWidget()
        self.image = gui.ImageWidget()

        grid = gui.VGrid(2, s * em)
        self.add_child(gui.Label("Focus Metric: "))
        self.add_child(self.metric_selector)
        self.add_child(self.plot)

    def set_metric_options(self, metrics):
        for metric in metrics:
            self.metric_selector.add_item(metric)

    def set_background_color(self, color: gui.Color):
        self.background_color = color

    def set_plot(self, image):
        pass
