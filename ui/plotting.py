from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, Arrow, VeeHead # type: ignore
from .config_ui import *
import numpy as np

# TODO auch camera mitscalieren, wenn Z geändert wird

class AcousticCameraPlot:
    def __init__(self, frame_width, frame_height, mic_positions, alphas, Z=2.0):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.mic_positions = mic_positions
        
        self.camera_cds = ColumnDataSource({'image_data': []})
        self.cds = ColumnDataSource(data=dict(x=[], y=[], s=[]))
        self.mic_cds = ColumnDataSource(data=dict(x=[], y=[]))
        
        self.arrow_x = None
        self.arrow_y = None
        
        self.alpha_x, self.alpha_y = alphas
        self.xmin, self.xmax, self.ymin, self.ymax = self.calculate_view_range(Z)
       
        self.fig = self._create_plot()

    def update_view_range(self, Z):
        self.xmin, self.xmax, self.ymin, self.ymax = self.calculate_view_range(Z)
        self.fig.x_range.start = self.xmin
        self.fig.x_range.end = self.xmax
        self.fig.y_range.start = self.ymin
        self.fig.y_range.end = self.ymax
        
    #TODO update view range video also

    def calculate_view_range(self, Z):
        xmax = Z * np.tan(self.alpha_x / 2)
        xmin = -xmax
        ymax = Z * np.tan(self.alpha_y / 2)
        ymin = -ymax
        return xmin, xmax, ymin, ymax

    def update_plot(self, model_data):
        self.cds.data = dict(x=model_data['x'], y=model_data['y'], s=model_data['s'])

    def update_camera_image(self, img):
        self.camera_cds.data['image_data'] = [img]

    def toggle_mic_visibility(self, visible):
        if visible:
            self.mic_cds.data = dict(x=self.mic_positions[0], y=self.mic_positions[1])
        else:
            self.mic_cds.data = dict(x=[], y=[])

    def toggle_origin_visibility(self, visible):
        if self.arrow_x and self.arrow_y:
            self.arrow_x.visible = visible
            self.arrow_y.visible = visible
            
    def _create_base_fig(self):
        fig = figure(width=self.frame_width, 
                     height=self.frame_height, 
                     x_range=(self.xmin, self.xmax), 
                     y_range=(self.ymin, self.ymax),
                     output_backend='webgl',
                     aspect_ratio=1)
        
        fig.image_rgba(image='image_data', 
                       x=self.xmin, 
                       y=self.ymin, 
                       dw=(self.xmax-self.xmin), 
                       dh=(self.ymax-self.ymin), 
                       source=self.camera_cds, 
                       alpha=VIDEOALPHA)
        
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
        
        self.arrow_x = Arrow(end=VeeHead(size=ORIGINHEADSIZE,fill_color=ORIGINCOLOR, line_color=ORIGINCOLOR), 
                             x_start=0, 
                             y_start=0, 
                             x_end=ORIGINLENGTH, 
                             y_end=0, 
                             line_width=ORIGINLINEWIDTH,
                             line_color=ORIGINCOLOR)
        fig.add_layout(self.arrow_x)
        
        self.arrow_y = Arrow(end=VeeHead(size=ORIGINHEADSIZE, fill_color=ORIGINCOLOR, line_color=ORIGINCOLOR), 
                             x_start=0, 
                             y_start=0, 
                             x_end=0, 
                             y_end=ORIGINLENGTH, 
                             line_width=ORIGINLINEWIDTH,
                             line_color=ORIGINCOLOR)
        fig.add_layout(self.arrow_y)
        
        fig.background_fill_color = PLOT_BACKGROUND_COLOR
        fig.border_fill_color = BACKGROUND_COLOR
        fig.outline_line_color = None 
        
        return fig
        

class AcousticCameraPlotModel(AcousticCameraPlot):
    def __init__(self, frame_width, frame_height, mic_positions, alphas, Z=2.0):
        super().__init__(frame_width, frame_height, mic_positions, alphas, Z)
        self.fig = self._create_plot()

    def _create_plot(self):
        fig = self._create_base_fig()
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
        
        return fig
    
    
class AcousticCameraPlotBeamforming(AcousticCameraPlot):
    def __init__(self, frame_width, frame_height, mic_positions, alphas, Z=2.0):
        super().__init__(frame_width, frame_height, mic_positions, alphas, Z)

        self.fig = self._create_plot()

    def _create_plot(self):
        fig = self._create_base_fig()
        # TODO Hier contour plot hinzufügen
        
        return fig


class StreamPlot():
    def __init__(self):
        self.cds_list = [ColumnDataSource(data=dict(x=[], y=[])) for _ in range(16)]
        self.fig = self._create_plot()
        
    def _create_plot(self):
        fig = figure(width=1100, 
                     height=300, 
                     output_backend='webgl')
        
        fig.line(x='x', y='y', source=self.cds_list[0], line_color=HIGHLIGHT_COLOR, alpha=0.5)
        
        fig.background_fill_color = PLOT_BACKGROUND_COLOR
        fig.border_fill_color = BACKGROUND_COLOR

        return fig
    
    def update_plot(self, stream_data):
        for i, cds in enumerate(self.cds_list):
            y_data = [row[i] for row in stream_data['y']]
            cds.data = dict(x=stream_data['x'], y=y_data)
            