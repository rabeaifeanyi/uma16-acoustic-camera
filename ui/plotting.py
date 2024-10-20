from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, Arrow, VeeHead, ColorBar, LinearColorMapper # type: ignore
from bokeh.palettes import Magma, Viridis256 # type: ignore
from bokeh.transform import linear_cmap # type: ignore
from .config_ui import *
import numpy as np
from scipy.spatial.distance import pdist, squareform # type: ignore


class AcousticCameraPlot:
    def __init__(self, frame_width, frame_height, mic_positions, alphas, threshold, scale_factor=1, Z=3.0, min_distance=1):
        
        # Set the frame width and height
        self.frame_width = int(frame_width * 1.1 * scale_factor)
        self.frame_height = int(frame_height * scale_factor)
        
        self.min_cluster_distance = 0
        
        self.cluster = 0
        
        self.Z = Z
        self.min_distance = min_distance
        
        # Array with microphone positions
        self.mic_positions = mic_positions
        
        # Threshold for the model data
        self.threshold = threshold
        
        # Data source for the camera image
        self.camera_cds = ColumnDataSource({'image_data': []})

        # Data source for the microphone positions
        self.mic_cds = ColumnDataSource(data=dict(x=[], y=[]))
        
        # Arrow for the origin
        self.arrow_x = None
        self.arrow_y = None
        
        # Camera view angles
        self.alpha_x, self.alpha_y = alphas
        
        # Calculate the view range
        self.xmin, self.xmax, self.ymin, self.ymax = self.calculate_view_range(Z)
        
        # Point sizes for the model data
        # TODO: sinnvolle Werte finden
        self.min_point_size, self.max_point_size = 2, 15
        
        # Data sources for the model data
        self.model_cds = ColumnDataSource(data=dict(x=[], y=[], z=[], s=[], sizes=[]))
        
        # Data source for the beamforming data
        self.beamforming_cds = ColumnDataSource({'beamformer_data': []})    
        
        self.x_min, self.y_min = -1.5, -1.5
        self.x_max, self.y_max = 1.5, 1.5
        self.dx = self.x_max - self.x_min
        self.dy = self.y_max - self.y_min
        
        self.bar_low, self.bar_high = 0.0001, 0.01
        
        # Create the plot
        self.fig = self._create_plot()

    def update_view_range(self, Z):
        self.xmin, self.xmax, self.ymin, self.ymax = self.calculate_view_range(Z)
        self.fig.x_range.start = self.xmin
        self.fig.x_range.end = self.xmax
        self.fig.y_range.start = self.ymin
        self.fig.y_range.end = self.ymax
        
    def update_threshold(self, threshold):
        self.threshold = threshold
        
    def update_cluster_distance(self, distance):
        self.min_cluster_distance = distance

    def calculate_view_range(self, Z):
        xmax = Z * np.tan(self.alpha_x / 2)
        xmin = -xmax
        ymax = Z * np.tan(self.alpha_y / 2)
        ymin = -ymax
        return xmin, xmax, ymin, ymax

    def update_plot_model(self, model_data):
        self.model_renderer.visible = True
        self.beamforming_renderer.visible = False
        
        x = np.array(model_data['x'])
        y = np.array(model_data['y'])
        z = np.array(model_data['z'])
        s = np.array(model_data['s'])
        
        if len(z) > 0:
            z_clipped = np.clip(z, self.min_distance, self.Z)
            
            z_norm = (z_clipped - self.min_distance) / (self.Z - self.min_distance)
            z_inverted = 1 - z_norm 
            
            sizes = self.min_point_size + z_inverted * (self.max_point_size - self.min_point_size)
        else:
            sizes = []
        
        mask = s >= self.threshold
        
        x, y, z, s, sizes = x[mask], y[mask], z[mask], s[mask], sizes[mask]
        
        if self.cluster:
            x, y, z = self.cluster_points(x, y, z)

        self.model_cds.data = dict(x=x, y=y, z=z, s=s, sizes=sizes)
        
    def cluster_points(self, x_list, y_list, z_list):
        points = np.array(list(zip(x_list, y_list, z_list)))
        dist_matrix = squareform(pdist(points))
        close_points = dist_matrix < self.min_cluster_distance
        
        groups = []
        visited = np.zeros(len(points), dtype=bool)

        for i in range(len(points)):
            if not visited[i]:
                group_indices = np.where(close_points[i])[0]
                group_points = points[group_indices]
                group_mean = np.mean(group_points, axis=0)
                groups.append(group_mean)
                visited[group_indices] = True

        groups = np.array(groups)
        x_result = groups[:, 0]
        y_result = groups[:, 1]
        z_result = groups[:, 2]
        
        return x_result, y_result, z_result 
    
    def update_plot_beamforming(self, results):
        self.model_renderer.visible = False
        self.beamforming_renderer.visible = True
        self.beamforming_cds.data['beamformer_data'] = results['results']

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
    
    def _create_plot(self):
        fig = self._create_base_fig()
        
        self.color_mapper = linear_cmap('s', Viridis256, self.bar_low, self.bar_high)
        
        # Renderer für Modell-Daten
        self.model_renderer = fig.scatter(
            x='x', 
            y='y',
            marker='circle', 
            size='sizes', 
            color=self.color_mapper,
            alpha=DOTALPHA, 
            source=self.model_cds
        )

        self.b_color_mapper = LinearColorMapper(palette=Viridis256, low=0, high=100)
       
        # Renderer für Beamforming-Daten
        self.beamforming_renderer = fig.image(
            image='beamformer_data',
            x=self.x_min,
            y=self.y_min,
            dw=self.dx,
            dh=self.dy,
            source=self.beamforming_cds,
            color_mapper=self.b_color_mapper,
            level='image',
            alpha=0.6
        )
        
        color_bar = ColorBar(
            color_mapper=self.color_mapper['transform'], 
            label_standoff=12, 
            width=8, 
            location=(0, 0),
            background_fill_color=PLOT_BACKGROUND_COLOR
        )

        fig.add_layout(color_bar, 'right')  
        
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
            