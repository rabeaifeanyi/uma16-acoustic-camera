import threading
from bokeh.layouts import column, layout, row
from bokeh.models import Div, CheckboxGroup, RadioButtonGroup, TextInput, Button # type: ignore
from bokeh.plotting import curdoc
from .plotting import AcousticCameraPlot, StreamPlot
from .config_ui import *


class Dashboard:
    def __init__(self, video_stream, processor, mic_array_config, estimation_update_interval, beamforming_update_interval, camera_update_interval, stream_update_interval, threshold, alphas, z, min_distance):

        # Angles of the camera view
        self.alphas = alphas
        
        # Video stream object
        self.video_stream = video_stream

        # Data Processor object, contains model and beamforming
        self.processor = processor
        
        # Setting model and beamforming threads to None in accordance with the start and stop logic
        self.model_thread = None
        self.beamforming_thread = None
        
        # Method for processing the data, 0 is Deep Learning, 1 is Beamforming
        self.method = 0 # Default is Deep Learning
        
        # Setting up the acoustic camera plot
        self.acoustic_camera_plot = AcousticCameraPlot(
                                        frame_width=video_stream.frame_width,
                                        frame_height=video_stream.frame_height,
                                        mic_positions=mic_array_config.mic_positions(),
                                        alphas=self.alphas,
                                        threshold=threshold,
                                        Z=z,
                                        min_distance=min_distance
                                    )    
        
        # Setting up the stream plot
        self.stream_plot = StreamPlot()
        
        # Setting up the update intervals
        self.estimation_update_interval = estimation_update_interval
        self.beamforming_update_interval = beamforming_update_interval
        self.camera_update_interval = camera_update_interval
        self.stream_update_interval = stream_update_interval
        self.overflow_update_interval = estimation_update_interval
        
        # Frequency input field
        self.f_input = TextInput(value=str(self.processor.frequency), title="Frequency (Hz)")

        # Overflow status text
        self.overflow_status = Div(text="Overflow Status: Unknown", width=300, height=30)
        
        # Switching between Deep Learning and Beamforming
        self.method_selector = RadioButtonGroup(labels=["Deep Learning", "Beamforming"], active=0)  # 0 is "Deep Learning" as default
        
        # Callbacks
        self.camera_view_callback = None
        self.estimation_callback = None
        self.beamforming_callback = None
        self.stream_callback = None
        self.overflow_callback = None

        # Setting up the layout
        self.setup_layout()
        
        # Setting up the callbacks
        self.setup_callbacks()
        
    def get_layout(self):
        """Return the layout of the dashboard
        """
        return self.dashboard_layout

    def setup_layout(self):
        """Setup the layout of the dashboard
        """
        # Styling for the sidebar
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
        
        # Header "Acoustic Camera"
        header = Div(text=f"<h1 style='color:{FONTCOLOR}; font-family:{FONT}; margin-left: 320px;'>Acoustic Camera</h1>", 
                     margin=(20, 0, 0, 0))
        
        # Checkboxes for origin and mic-geom visibility
        self.checkbox_group = CheckboxGroup(labels=["Show Microphone Geometry", "Show Origin"], active=[0, 1])
        
        # Mode Selector (Acoustic Camera vs. Stream)
        mode_selector = RadioButtonGroup(labels=["Acoustic Camera", "Stream"], active=0)
        
        # Measurement button
        # Ideas: Color change when measurement is active, change label to "Stop Measurement"
        # Problem: When stop is pressed to quickly and model has not properly started, error occurs
        # Possible solution: Add a check if model_thread is None before stopping
        self.measurement_button = Button(label="Start Messung")
        
        # Grouping relevant UI elements into a single container (sidebar_section)
        self.sidebar_section = column(
            self.f_input,
            self.checkbox_group,
            self.method_selector,
            self.measurement_button,
            self.overflow_status
        )
        
        # Sidebar layout
        sidebar = column(
            Div(text=f"{sidebar_style}<div id='sidebar'></div>", width=SIDEBAR_WIDTH),
            mode_selector,
            self.sidebar_section
        )
        
        # Plot visibility control
        self.stream_plot.fig.visible = False
        
        # Content layout of Page
        content_layout = column(
            header,
            self.acoustic_camera_plot.fig,
            self.stream_plot.fig,
            sizing_mode="stretch_both",
            margin=(0, 320, 0, 0) 
        )

        # Main dashboard layout
        self.dashboard_layout = layout(
            row(sidebar, content_layout),
            sizing_mode="stretch_both",
            background=BACKGROUND_COLOR,
            margin=(0, 0, 0, 0)
        )

        # Callbacks
        self.checkbox_group.on_change("active", self.toggle_visibility)
        mode_selector.on_change('active', self.toggle_plot_visibility)
        self.method_selector.on_change('active', self.toggle_method)

    def setup_callbacks(self):
        """Setup the callbacks for the dashboard
        """
        # Callbacks for the frequency input field
        self.f_input.on_change("value", self.update_frequency)
        
        # Callbacks for the measurement button
        self.measurement_button.on_click(self.toggle_measurement)

        # Start the acoustic camera plot
        self.start_acoustic_camera_plot()

    def toggle_method(self, attr, old, new):
        """Callback for the method selector"""
        # Stop the current method
        
        self.stop_measurement()
        
        # Remove the periodic callbacks
        if self.estimation_callback is not None:
            curdoc().remove_periodic_callback(self.estimation_callback)
            self.estimation_callback = None
        if self.beamforming_callback is not None:
            curdoc().remove_periodic_callback(self.beamforming_callback)
            self.beamforming_callback = None
        
        # Start the new method
        if new == 0:
            print("Wechsel zu Deep Learning")
            self.method = 0
            self.estimation_callback = curdoc().add_periodic_callback(
                self.update_estimations, self.estimation_update_interval)
        elif new == 1:
            print("Wechsel zu Beamforming")
            self.method = 1
            self.beamforming_callback = curdoc().add_periodic_callback(
                self.update_beamforming, self.beamforming_update_interval)
            
    def stop_measurement(self):
        """Stop the current measurement"""
        if self.model_thread is not None:
            self.stop_model()
            self.measurement_button.label = start_text
        if self.beamforming_thread is not None:
            self.stop_beamforming()
            self.measurement_button.label = start_text

                
    def toggle_measurement(self):
        """Callback für den Messungs-Button, startet oder stoppt die Messung"""
        if self.method == 0:
            if self.model_thread is None:
                self.start_model()
                self.measurement_button.label = stop_text
            else:
                self.stop_model()
                self.measurement_button.label = start_text
        elif self.method == 1:
            if self.beamforming_thread is None:
                self.start_beamforming()
                self.measurement_button.label = stop_text
            else:
                self.stop_beamforming()
                self.measurement_button.label = start_text

        
    def update_frequency(self, attr, old, new):
        """Callback for the frequency input field
        """
        try:
            f = float(new)
            self.processor.update_frequency(f)
        except ValueError:
            pass
    
    def update_overflow_status(self):
        """Update the overflow status text
        """
        overflow = self.processor.dev.overflow
        status_text = f"Overflow Status: {overflow}"
        self.overflow_status.text = status_text

    # Acoustic Camera Plot
    def start_acoustic_camera_plot(self):
        self.stop_stream_plot()
        self.video_stream.start()
        
        # Deep Learning
        if self.method == 0:
            self.stop_beamforming()
            self.acoustic_camera_plot.beamforming_renderer.visible = False
            self.acoustic_camera_plot.model_renderer.visible = True
        
        # Beamforming
        elif self.method == 1:
            self.stop_model()
            self.acoustic_camera_plot.model_renderer.visible = False
            self.acoustic_camera_plot.beamforming_renderer.visible = True

        if self.camera_view_callback is None:
            self.camera_view_callback = curdoc().add_periodic_callback(self.update_camera_view, self.camera_update_interval)
        
    def stop_acoustic_camera_plot(self):
        """Stop periodic callbacks for the acoustic camera plot"""
        if self.camera_view_callback is not None:
            curdoc().remove_periodic_callback(self.camera_view_callback)
            self.camera_view_callback = None

        if self.estimation_callback is not None:
            curdoc().remove_periodic_callback(self.estimation_callback)
            self.estimation_callback = None

        if self.beamforming_callback is not None:
            curdoc().remove_periodic_callback(self.beamforming_callback)
            self.beamforming_callback = None
            
        self.stop_measurement()
        self.video_stream.stop()
            
    # Stream Plot
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

    # Sidebar methods
    def toggle_plot_visibility(self, attr, old, new):
        self.stop_measurement()
        self.measurement_button.label = "Start Messung"
        
        if new == 0:  # Acoustic Camera selected
            self.acoustic_camera_plot.fig.visible = True
            self.stream_plot.fig.visible = False
            self.start_acoustic_camera_plot()
            
            # Show all elements in the sidebar
            self.sidebar_section.visible = True

        elif new == 1:  # Stream selected
            self.acoustic_camera_plot.fig.visible = False
            self.stream_plot.fig.visible = True
            self.start_stream_plot()
            
            # Hide all elements in the sidebar
            self.sidebar_section.visible = False

    def toggle_mic_visibility(self, visible):
        self.acoustic_camera_plot.toggle_mic_visibility(visible)
            
    def toggle_origin_visibility(self, visible):
        self.acoustic_camera_plot.toggle_origin_visibility(visible)
            
    def toggle_visibility(self, attr, old, new):
        self.toggle_mic_visibility(0 in new)
        self.toggle_origin_visibility(1 in new)
    
    # Camera View
    def update_camera_view(self):
        img = self.video_stream.get_frame()
        if img is not None:
            self.acoustic_camera_plot.update_camera_image(img)

    # Deep Learning
    def start_model(self):
        if self.model_thread is None:
            self.model_thread = threading.Thread(target=self.processor.start_model, daemon=True)
            self.model_thread.start()
        
        if self.estimation_callback is None:
            self.estimation_callback = curdoc().add_periodic_callback(self.update_estimations, self.estimation_update_interval)
            
        if self.overflow_callback is None:
            self.overflow_callback = curdoc().add_periodic_callback(self.update_overflow_status, self.overflow_update_interval)

                        
    def stop_model(self):
        if self.model_thread is not None:
            self.processor.stop_model()
            self.model_thread.join()
            self.model_thread = None
            
        if self.estimation_callback is not None:
            curdoc().remove_periodic_callback(self.estimation_callback)
            self.estimation_callback = None
            
        if self.overflow_callback is not None:
            curdoc().remove_periodic_callback(self.overflow_callback)
            self.overflow_callback = None
            
    def update_estimations(self):
        model_data = self.processor.results
        self.acoustic_camera_plot.update_plot_model(model_data)
         
    # Beamforming   
    def start_beamforming(self):
        if self.beamforming_thread is None:
            self.beamforming_thread = threading.Thread(target=self.processor.start_beamforming, daemon=True)
            self.beamforming_thread.start()
        
        if self.beamforming_callback is None:
            self.beamforming_callback = curdoc().add_periodic_callback(self.update_beamforming, self.beamforming_update_interval)
            
        if self.overflow_callback is None:
            self.overflow_callback = curdoc().add_periodic_callback(self.update_overflow_status, self.overflow_update_interval)
            
    def stop_beamforming(self):
        if self.beamforming_thread is not None:
            self.beamforming_thread.join()
            self.beamforming_thread = None
            
        if self.beamforming_callback is not None:
            curdoc().remove_periodic_callback(self.beamforming_callback)
            self.beamforming_callback = None
            
        if self.overflow_callback is not None:
            curdoc().remove_periodic_callback(self.overflow_callback)
            self.overflow_callback = None

    def update_beamforming(self):
        beamforming_data = self.processor.results
        self.acoustic_camera_plot.update_plot_beamforming(beamforming_data)
    
    # Stream
    def update_stream(self):
        stream_data = self.processor.get_uma_data()
        self.stream_plot.update_plot(stream_data)

