from bokeh.layouts import column, layout, row, Spacer
from bokeh.models import Div, CheckboxGroup # type: ignore
from bokeh.plotting import curdoc
from .plot import create_plot, update_plot # type: ignore
from design import *


ESTIMATION_UPDATE_INTERVAL = 1000
CAMERA_UPDATE_INTERVAL = 100


def create_dashboard(video_stream, model_processor, config):
    """Creates the dashboard layout for the application.

    Args:
        video_stream (VideoStream): Instance of the video stream.
        model_processor (ModelProcessor): Instance of the model processor.
        config (ConfigUMA): Configuration for the UMA-16 microphone array.

    Returns:
        layout: Bokeh layout containing the entire dashboard.
    """

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
    sidebar = Div(text=f"{sidebar_style}<div id='sidebar'></div>", width=SIDEBAR_WIDTH)
    header = Div(text=f"<h1 style='color:{FONTCOLOR}; font-family:{FONT}; margin-left: 320px;'>Acoustic Camera</h1>", margin=(20, 0, 0, 0))

    frame_width = video_stream.frame_width
    frame_height = video_stream.frame_height
    mic_positions = config.mic_positions()

    fig, cds, mic_cds, cameraCDS = create_plot(frame_width, frame_height, mic_positions)
    fig.width = FIG_WIDTH
    fig.height = FIG_HEIGHT

    content_layout = column(
        header,
        fig,
        sizing_mode="stretch_both",
        margin=(0, 320, 0, 0) 
    )

    dashboard_layout = layout(
        row(sidebar, content_layout),
        sizing_mode="stretch_both",
        background=BACKGROUND_COLOR,
        margin=(0, 0, 0, 0)
    )

    def update_camera_view():
        img = video_stream.get_frame()
        if img is not None:
            cameraCDS.data['image_data'] = [img]

    def update_estimations():
        model_data = model_processor.get_uma16_dummy_data()
        update_plot(cds, model_data)

    doc = curdoc()
    doc.add_periodic_callback(update_estimations, ESTIMATION_UPDATE_INTERVAL)
    doc.add_periodic_callback(update_camera_view, CAMERA_UPDATE_INTERVAL)

    return dashboard_layout
