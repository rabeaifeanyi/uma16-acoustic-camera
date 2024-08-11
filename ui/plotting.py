from bokeh.plotting import figure
from bokeh.models import ColumnDataSource # type: ignore
from design import *


# Visual Range = Estimation -> TODO make this configurable
XMIN = -2.5
XMAX = 2.5
YMIN = -1.75
YMAX = 1.75
Z = 2.0

class AcousticCameraPlot:
    def __init__(self, frame_width, frame_height, mic_positions):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.mic_positions = mic_positions

        self.cameraCDS = ColumnDataSource({'image_data': []})
        self.cds = ColumnDataSource(data=dict(x=[], y=[], s=[]))
        self.mic_cds = ColumnDataSource(data=dict(x=[], y=[]))

        self.fig = self._create_plot()

    def _create_plot(self):
        fig = figure(width=self.frame_width, 
                     height=self.frame_height, 
                     x_range=(XMIN, XMAX), 
                     y_range=(YMIN, YMAX),
                     output_backend='webgl')
        
        fig.image_rgba(image='image_data', 
                       x=XMIN, 
                       y=YMIN, 
                       dw=(XMAX-XMIN), 
                       dh=(YMAX-YMIN), 
                       source=self.cameraCDS, 
                       alpha=VIDEOALPHA)
        
        fig.scatter(x='x', 
                    y='y', 
                    legend_label='Strength of Source', 
                    marker='circle', 
                    size='s', 
                    color=SHADOWCOLOR, 
                    alpha=SHADOWALPHA, 
                    line_color=None,
                    source=self.cds)
        
        fig.scatter(x='x', 
                    y='y', 
                    legend_label='Sound Source', 
                    marker='circle', 
                    size=DOTSIZE, 
                    color=DOTCOLOR, 
                    alpha=DOTALPHA, 
                    source=self.cds)
        
        self.mic_cds.data = dict(x=self.mic_positions[0], y=self.mic_positions[1])
        fig.scatter(x='x', 
                    y='y', 
                    legend_label='Microphones', 
                    marker='circle', 
                    size=MICSIZE, 
                    color=MICCOLOR, 
                    line_color=MICLINECOLOR,
                    alpha=MICALPHA, 
                    source=self.mic_cds)
        
        fig.background_fill_color = PLOT_BACKGROUND_COLOR
        fig.border_fill_color = BACKGROUND_COLOR
        fig.outline_line_color = None 
        
        return fig

    def update_plot(self, model_data):
        self.cds.data = dict(x=model_data['x'], y=model_data['y'], s=model_data['s'])

    def update_camera_image(self, img):
        self.cameraCDS.data['image_data'] = [img]

    def toggle_mic_visibility(self, visible):
        if visible:
            self.mic_cds.data = dict(x=self.mic_positions[0], y=self.mic_positions[1])
        else:
            self.mic_cds.data = dict(x=[], y=[])
    
# mic_positions:
# [[ 0.021  0.063  0.021  0.063  0.021  0.063  0.021  0.063 -0.063 -0.021 -0.063 -0.021 -0.063 -0.021 -0.063 -0.021]
#  [-0.063 -0.063 -0.021 -0.021  0.021  0.021  0.063  0.063  0.063  0.063  0.021  0.021 -0.021 -0.021 -0.063 -0.063]
#  [ 0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.   ]]
