from bokeh.layouts import column, layout
from bokeh.models import Div, Slider, Button # type: ignore
from bokeh.plotting import curdoc
from .plot import create_plot, update_plot # type: ignore


def create_dashboard(video_stream, model_processor, config):
    """Creates the dashboard layout for the application.

    Args:
        video_stream (VideoStream): Instance of the video stream.
        model_processor (ModelProcessor): Instance of the model processor.
        config (ConfigUMA): Configuration for the UMA-16 microphone array.

    Returns:
        layout: Bokeh layout containing the entire dashboard.
    """

    # Set up the header
    header = Div(text="<h1>Acoustic Camera</h1>", width=800)

    # Set up the sidebar with controls
    slider = Slider(start=0.1, end=2.0, value=1.0, step=0.1, title="A Slider")

    sidebar = column(slider)

    # Set up the main plot area
    frame_width = video_stream.frame_width
    frame_height = video_stream.frame_height
    mic_positions = config.mic_positions()

    fig, cds, mic_cds, cameraCDS = create_plot(frame_width, frame_height, mic_positions)

    # Define the layout structure
    dashboard_layout = layout([
        [header],
        [sidebar, fig]
    ])

    # Callbacks for interactive widgets
    def update_camera_view():
        img = video_stream.get_frame()
        if img is not None:
            cameraCDS.data['image_data'] = [img]

    def update_estimations():
        model_data = model_processor.get_uma16_dummy_data()
        update_plot(cds, model_data)

    def update_slider(attr, old, new):
        print(f"Slider value changed from {old} to {new}")

    # Link callbacks to widgets
    slider.on_change('value', update_slider)

    # Add periodic updates for camera and estimations
    doc = curdoc()
    doc.add_periodic_callback(update_estimations, 1000)
    doc.add_periodic_callback(update_camera_view, 100)

    return dashboard_layout
