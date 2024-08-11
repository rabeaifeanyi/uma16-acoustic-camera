from bokeh.layouts import column, layout, row
from bokeh.models import Div, CheckboxGroup # type: ignore
from bokeh.plotting import curdoc
from .plotting import AcousticCameraPlot
from design import *


ESTIMATION_UPDATE_INTERVAL = 1000
CAMERA_UPDATE_INTERVAL = 100

class Dashboard:
    def __init__(self, video_stream, model_processor, config):
        self.video_stream = video_stream
        self.model_processor = model_processor
        self.acoustic_plot = AcousticCameraPlot(
            frame_width=video_stream.frame_width,
            frame_height=video_stream.frame_height,
            mic_positions=config.mic_positions()
        )
        
        self.setup_layout()
        self.setup_callbacks()

    def setup_layout(self):
        sidebar_style = """
        <style>
            #sidebar {
                position: fixed;
                top: 0;
                left: 0;
                height: 100%;
                width: 300px;
                background-color: #ecf0f1;
                padding: 20px;
                box-sizing: border-box;
            }
        </style>
        """
        header = Div(text=f"<h1 style='color:{FONTCOLOR}; font-family:{FONT}; margin-left: 320px;'>Acoustic Camera</h1>", margin=(20, 0, 0, 0))

        checkbox_group = CheckboxGroup(labels=["Show Microphone Geometry"], active=[0])

        sidebar = column(Div(text=f"{sidebar_style}<div id='sidebar'></div>", width=SIDEBAR_WIDTH),
                         checkbox_group)
        
        content_layout = column(
            header,
            self.acoustic_plot.fig,
            sizing_mode="stretch_both",
            margin=(0, 320, 0, 0) 
        )

        self.dashboard_layout = layout(
            row(sidebar, content_layout),
            sizing_mode="stretch_both",
            background=BACKGROUND_COLOR,
            margin=(0, 0, 0, 0)
        )

        checkbox_group.on_change("active", self.toggle_mic_visibility)

    def setup_callbacks(self):
        curdoc().add_periodic_callback(self.update_estimations, ESTIMATION_UPDATE_INTERVAL)
        curdoc().add_periodic_callback(self.update_camera_view, CAMERA_UPDATE_INTERVAL)

    def toggle_mic_visibility(self, attr, old, new):
        if 0 in new:
            self.acoustic_plot.toggle_mic_visibility(True)
        else:
            self.acoustic_plot.toggle_mic_visibility(False)

    def update_camera_view(self):
        img = self.video_stream.get_frame()
        if img is not None:
            self.acoustic_plot.update_camera_image(img)

    def update_estimations(self):
        model_data = self.model_processor.get_uma16_dummy_data()
        self.acoustic_plot.update_plot(model_data)

    def get_layout(self):
        return self.dashboard_layout
