import threading
from bokeh.layouts import column, layout, row
from bokeh.models import Div, CheckboxGroup, RadioButtonGroup, TextInput # type: ignore
from bokeh.plotting import curdoc
from .plotting import AcousticCameraPlot, StreamPlot
from .config_ui import *

# TODO scaling of Video Stream and Plot

class Dashboard:
    """ Dashboard class for the acoustic camera application """
    
    def __init__(self, video_stream, model_processor, mic_array_config, estimation_update_interval, camera_update_interval, stream_update_interval, alphas, dummy=False, Z=2.0):
        # general setup
        self.dummy = dummy
        
        # Camera setup
        self.Z = Z
        self.alphas = alphas
        self.video_stream = video_stream
        
        # Model setup
        self.model_processor = model_processor
        self.model_thread = None
        
        # Plot setup
        self.acoustic_camera_plot = AcousticCameraPlot(
                                        frame_width=video_stream.frame_width,
                                        frame_height=video_stream.frame_height,
                                        mic_positions=mic_array_config.mic_positions(),
                                        alphas = self.alphas
                                    )    
        self.stream_plot = StreamPlot()
        self.estimation_update_interval = estimation_update_interval
        self.camera_update_interval = camera_update_interval
        self.stream_update_interval = stream_update_interval
        
        # UI setup
        self.z_input = TextInput(value=str(self.Z), title="Scale Z")
        
        # Callbacks
        self.camera_view_callback = None
        self.estimation_callback = None
        self.stream_callback = None

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
        
        checkbox_group = CheckboxGroup(labels=["Show Microphone Geometry", "Show Origin"], active=[0, 1])
        
        plot_selector = RadioButtonGroup(labels=["Acoustic Camera", "Stream"], active=0)
        
        sidebar = column(
            Div(text=f"{sidebar_style}<div id='sidebar'></div>", width=SIDEBAR_WIDTH),
            self.z_input, 
            checkbox_group,
            plot_selector
        )
        
        self.stream_plot.fig.visible = False
        
        content_layout = column(
            header,
            self.acoustic_camera_plot.fig,
            self.stream_plot.fig,
            sizing_mode="stretch_both",
            margin=(0, 320, 0, 0) 
        )

        self.dashboard_layout = layout(
            row(sidebar, content_layout),
            sizing_mode="stretch_both",
            background=BACKGROUND_COLOR,
            margin=(0, 0, 0, 0)
        )

        checkbox_group.on_change("active", self.toggle_visibility)
        plot_selector.on_change('active', self.toggle_plot_visibility)
        
    def setup_callbacks(self):
        self.z_input.on_change("value", self.update_scale_z)
        self.start_acoustic_camera_plot()

    def update_scale_z(self, attr, old, new):
        try:
            Z = float(new)
            self.acoustic_camera_plot.update_view_range(Z)
        except ValueError:
            pass

    def start_acoustic_camera_plot(self):
        self.stop_stream_plot()
        self.video_stream.start()
        self.start_model_in_background() #?

        if self.camera_view_callback is None:
            self.camera_view_callback = curdoc().add_periodic_callback(self.update_camera_view, self.camera_update_interval)
        
        if self.estimation_callback is None:
            self.estimation_callback = curdoc().add_periodic_callback(self.update_estimations, self.estimation_update_interval)

    def stop_acoustic_camera_plot(self):
        """Stop periodic callbacks for the acoustic camera plot"""
        if self.camera_view_callback is not None:
            curdoc().remove_periodic_callback(self.camera_view_callback)
            self.camera_view_callback = None

        if self.estimation_callback is not None:
            curdoc().remove_periodic_callback(self.estimation_callback)
            self.estimation_callback = None

        self.video_stream.stop()
        self.stop_model()

    def start_stream_plot(self):
        """Start periodic callbacks for the stream plot"""
        self.stop_acoustic_camera_plot()

        if self.stream_callback is None:
            self.stream_callback = curdoc().add_periodic_callback(self.update_stream, self.stream_update_interval)
    
    def stop_stream_plot(self):
        """Stop periodic callbacks for the stream plot"""
        if self.stream_callback is not None:
            curdoc().remove_periodic_callback(self.stream_callback)
            self.stream_callback = None

    def toggle_plot_visibility(self, attr, old, new):
        if new == 0:
            self.acoustic_camera_plot.fig.visible = True
            self.stream_plot.fig.visible = False
            self.start_acoustic_camera_plot()
        elif new == 1:
            self.acoustic_camera_plot.fig.visible = False
            self.stream_plot.fig.visible = True
            self.start_stream_plot()

    def toggle_mic_visibility(self, visible):
        self.acoustic_camera_plot.toggle_mic_visibility(visible)
            
    def toggle_origin_visibility(self, visible):
        self.acoustic_camera_plot.toggle_origin_visibility(visible)
            
    def toggle_visibility(self, attr, old, new):
        self.toggle_mic_visibility(0 in new)
        self.toggle_origin_visibility(1 in new)
        
    def start_model_in_background(self):
        """Start the model in a separate thread."""
        if self.model_thread is None:
            if self.dummy:
                self.model_thread = threading.Thread(target=self.model_processor.start_dummy_model, daemon=True)
            else:
                self.model_thread = threading.Thread(target=self.model_processor.start_model, daemon=True)
            self.model_thread.start()
            
    def stop_model(self):
        """Stop the model thread."""
        if self.model_thread is not None:
            self.model_processor.stop_model()
            self.model_thread.join()
            self.model_thread = None
        
    def update_camera_view(self):
        img = self.video_stream.get_frame()
        if img is not None:
            self.acoustic_camera_plot.update_camera_image(img)

    def update_estimations(self):
        model_data = self.model_processor.results
        print(model_data)
        self.acoustic_camera_plot.update_plot(model_data)
        
    def update_stream(self):
        stream_data = self.model_processor.get_uma_data()
        self.stream_plot.update_plot(stream_data)

    def get_layout(self):
        return self.dashboard_layout
