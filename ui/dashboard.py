import threading
import numpy as np
import datetime
from bokeh.layouts import column, layout, row
from bokeh.models import Div, CheckboxGroup, RadioButtonGroup, TextInput, Button, ColumnDataSource # type: ignore
from bokeh.plotting import curdoc, figure
from .plotting import AcousticCameraPlot, StreamPlot
from .config_ui import *


class Dashboard:
    def __init__(self, video_stream, processor, mic_array_config, 
                 estimation_update_interval, beamforming_update_interval, camera_update_interval, stream_update_interval, 
                 threshold, alphas, scale_factor, z, min_distance, camera_on=True):

        # Angles of the camera view
        self.alphas = alphas
        self.scale_factor = scale_factor    
        self.actual_width = int(video_stream.frame_width * 1.1 * scale_factor)
        
        # Video stream object
        self.video_stream = video_stream
        self.camera_on = camera_on
        
        self.start_time = 0
        
        self.frame_width, self.frame_height = video_stream.frame_width, video_stream.frame_height

        # Data Processor object, contains model and beamforming
        self.processor = processor
        
        # Setting model and beamforming threads to None in accordance with the start and stop logic
        self.model_thread = None
        self.beamforming_thread = None
        
        # Method for processing the data, 0 is Deep Learning, 1 is Beamforming
        self.method = 0 # Default is Deep Learning
        
        # Setting up the acoustic camera plot
        self.acoustic_camera_plot = AcousticCameraPlot(
                                        frame_width=self.frame_width,
                                        frame_height=self.frame_height,
                                        mic_positions=mic_array_config.mic_positions(shifted=False),
                                        alphas=self.alphas,
                                        threshold=threshold,
                                        Z=z,
                                        scale_factor=scale_factor,
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
        
        self.real_x, self.real_y, self.real_z = 1.0, 1.0, z
        
        # Frequency input field
        self.f_input = TextInput(value=str(self.processor.frequency), title="Frequency (Hz)")

        self.x_input = TextInput(value=str(self.real_x), title="Real X")
        self.y_input = TextInput(value=str(self.real_y), title="Real Y")
        self.z_input = TextInput(value=str(self.real_z), title="Real Z")

        # CSM Block size input field
        self.csm_block_size_input = TextInput(value=str(self.processor.csm_block_size), title="CSM Block Size")
        
        # CSM minimum queue size input field
        self.min_queue_size_input = TextInput(value=str(self.processor.min_queue_size), title="Minimum Queue Size")
        
        # Threshold input field
        self.threshold_input = TextInput(value=str(self.acoustic_camera_plot.threshold), title="Threshold")
        
        # Cluster distance input field
        self.cluster_distance_input = TextInput(value=str(self.acoustic_camera_plot.min_cluster_distance), title="Cluster Distance")
                                                         
        # Overflow status text
        self.overflow_status = Div(text="Overflow Status: Unknown", width=300, height=30)
        
        # Coordinates
        self.coordinates_display = Div(text="", width=300, height=100)
        
        # Level display
        self.level_display = Div(text="", width=300, height=100)
        
        # Plot of the deviation of the estimated position
        self.deviation_cds = ColumnDataSource(data=dict(time=[], x_deviation=[], y_deviation=[], z_deviation=[]))
        
        self.cluster_results = RadioButtonGroup(labels=[" ", "Cluster Results"], active=self.acoustic_camera_plot.cluster) 
        
        # Switching between Deep Learning and Beamforming
        self.method_selector = RadioButtonGroup(labels=["Deep Learning", "Beamforming"], active=self.method)  # 0 is "Deep Learning" as default
        
        # Callbacks
        self.camera_view_callback = None
        self.estimation_callback = None
        self.beamforming_callback = None
        self.stream_callback = None
        self.overflow_callback = None

        self._create_deviation_plot()
        
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
            self.method_selector,
            self.cluster_results,
            self.cluster_distance_input,
            self.x_input,
            self.y_input,
            self.z_input,
            self.f_input,
            self.csm_block_size_input,
            self.min_queue_size_input,
            self.threshold_input,
            self.checkbox_group,
            self.measurement_button,
            self.overflow_status,
            self.coordinates_display,
            self.level_display
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
            self.deviation_plot,
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
        self.cluster_results.on_change('active', self.toggle_cluster)

    def setup_callbacks(self):
        """Setup the callbacks for the dashboard
        """
        self.x_input.on_change("value", self.update_real_x)
        self.y_input.on_change("value", self.update_real_y)
        self.z_input.on_change("value", self.update_real_z)
        
        # Callbacks for the frequency input field
        self.f_input.on_change("value", self.update_frequency)
        
        # Callbacks for the CSM Block size input field
        self.csm_block_size_input.on_change("value", self.update_csm_block_size)
        
        # Callbacks for the minimum number of CSMs in the buffer
        self.min_queue_size_input.on_change("value", self.update_min_queue_size)
        
        # Callbacks for the threshold input field
        self.threshold_input.on_change("value", self.update_threshold)
        
        # Callbacks for the min cluster distance input field
        self.cluster_distance_input.on_change("value", self.update_min_cluster_distance)
        
        # Callbacks for the measurement button
        self.measurement_button.on_click(self.start_measurement)

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
                self.update_beamforming_dot, self.beamforming_update_interval)
            
    def toggle_cluster(self, attr, old, new):
        """Callback for the cluster results selector"""
        self.acoustic_camera_plot.cluster = new
            
    def stop_measurement(self):
        """Stop the current measurement"""
        if self.model_thread is not None:
            self.stop_model()
            self.measurement_button.label = start_text
        if self.beamforming_thread is not None:
            self.stop_beamforming()
            self.measurement_button.label = start_text
                
    def start_measurement(self):
        """Callback für den Messungs-Button, startet oder stoppt die Messung"""
        if self.method == 0:
            if self.model_thread is None:
                self.snapshot_callback = curdoc().add_next_tick_callback(self.update_snapshot)
                self.start_model()
                self.start_time = datetime.datetime.now()
                self.measurement_button.label = stop_text
            else:
                self.stop_model()
                self.measurement_button.label = start_text
        elif self.method == 1:
            if self.beamforming_thread is None:
                self.snapshot_callback = curdoc().add_next_tick_callback(self.update_snapshot)
                self.start_beamforming()
                self.start_time = datetime.datetime.now()
                self.measurement_button.label = stop_text
            else:
                self.stop_beamforming()
                self.measurement_button.label = start_text
                
    def update_real_x(self, attr, old, new):
        """Callback for the real x input field
        """
        try:
            x = float(new)
            self.real_x = x
        except ValueError:
            pass
    
    def update_real_y(self, attr, old, new):
        """Callback for the real y input field
        """
        try:
            y = float(new)
            self.real_y = y
        except ValueError:
            pass
        
    def update_real_z(self, attr, old, new):
        """Callback for the real z input field
        """
        try:
            z = float(new)
            self.real_z = z
        except ValueError:
            pass
        
    def update_frequency(self, attr, old, new):
        """Callback for the frequency input field
        """
        try:
            f = float(new)
            self.processor.update_frequency(f)
        except ValueError:
            pass
        
    def update_csm_block_size(self, attr, old, new):
        """Callback for the csmblocksize input field
        """
        try:
            s = float(new)
            self.processor.update_csm_block_size(s)
        except ValueError:
            pass
        
    def update_min_queue_size(self, attr, old, new):
        """Callback for minqueuesize input field
        """
        try:
            s = float(new)
            self.processor.update_min_queue_size(s)
        except ValueError:
            pass
        
    def update_threshold(self, attr, old, new):
        """Callback for the threshold input field
        """
        try:
            t = float(new)
            self.acoustic_camera_plot.update_threshold(t)
        except ValueError:
            pass
        
    def update_min_cluster_distance(self, attr, old, new):
        """Callback for the min cluster distance input field
        """
        try:
            d = float(new)
            self.acoustic_camera_plot.update_min_cluster_distance(d)
        except ValueError:
            pass
        
    def update_overflow_status(self):
        """Update the overflow status text
        """
        overflow = self.processor.dev.overflow
        status_text = f"Overflow Status: {overflow}"
        self.overflow_status.text = status_text
        
    def update_snapshot(self):
        self.video_stream.take_snapshot()
        filename = self._get_result_filenames()
        self.video_stream.save_snapshot(filename)
        snapshot = self.video_stream.img    
        self.acoustic_camera_plot.update_camera_image(snapshot)
        
    def _create_deviation_plot(self):
        self.deviation_plot = figure(width=self.actual_width, height=250, title="Live Deviation Plot")
        self.deviation_plot.line(x='time', y='x_deviation', source=self.deviation_cds, color="blue", legend_label="X Deviation")
        self.deviation_plot.line(x='time', y='y_deviation', source=self.deviation_cds, color="green", legend_label="Y Deviation")
        self.deviation_plot.line(x='time', y='z_deviation', source=self.deviation_cds, color="red", legend_label="Z Deviation")
        self.deviation_plot.background_fill_color = BACKGROUND_COLOR
        self.deviation_plot.border_fill_color = BACKGROUND_COLOR
        
    def _get_result_filenames(self):
        """ Get the filenames for the results
        """
        current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        return self.processor.results_folder + f'/{current_time}.png'

    # Acoustic Camera Plot
    def start_acoustic_camera_plot(self):
        self.stop_stream_plot()
        
        if self.camera_on:
            self.video_stream.start()  
            
        if not self.camera_on:
            self.snapshot_callback = curdoc().add_next_tick_callback(self.update_snapshot)
        
        # Deep Learning
        if self.method == 0:
            self.stop_beamforming()
            #self.acoustic_camera_plot.beamforming_renderer.visible = False
            self.acoustic_camera_plot.model_renderer.visible = True
        
        # Beamforming
        elif self.method == 1:
            self.stop_model()
            self.acoustic_camera_plot.model_renderer.visible = False
            #self.acoustic_camera_plot.beamforming_renderer.visible = True
            #print(self.acoustic_camera_plot.beamforming_renderer.visible)

        if self.camera_on:
            if self.camera_view_callback is None:
                self.camera_view_callback = curdoc().add_periodic_callback(self.update_camera_view, self.camera_update_interval)

    def stop_acoustic_camera_plot(self):
        """Stop periodic callbacks for the acoustic camera plot"""
        
        if self.camera_on:
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
        if self.camera_on:
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
        self.video_stream.get_frame()
        self.acoustic_camera_plot.update_camera_image(self.video_stream.img)

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
        model_data = self.processor.get_results()
        self.acoustic_camera_plot.update_plot_model(model_data)
        
        x_vals = np.round(model_data['x'], 2)
        y_vals = np.round(model_data['y'], 2)
        z_vals = np.round(model_data['z'], 2)
        
        self.coordinates_display.text = f"X: {x_vals}<br>Y: {y_vals}<br>Z: {z_vals}"
        
        self.level_display.text = f"Level: {model_data['s']}"
        
        x_deviation = np.array(model_data['x']) - self.real_x
        y_deviation = np.array(model_data['y']) - self.real_y
        z_deviation = np.array(model_data['z']) - self.real_z
        
        if self.start_time != 0:
            elapsed_time = (datetime.datetime.now() - self.start_time).total_seconds()
        else:
            elapsed_time = 0 
        
        new_deviation_data = dict(
            time=[elapsed_time],
            x_deviation=[np.mean(x_deviation)],  # Durchschnittliche Abweichung
            y_deviation=[np.mean(y_deviation)],
            z_deviation=[np.mean(z_deviation)]
        )
    
        self.deviation_cds.stream(new_deviation_data, rollover=200)
            
    # Beamforming   
    def start_beamforming(self):
        if self.beamforming_thread is None:
            self.beamforming_thread = threading.Thread(target=self.processor.start_beamforming, daemon=True)
            self.beamforming_thread.start()
        
        if self.beamforming_callback is None:
            self.beamforming_callback = curdoc().add_periodic_callback(self.update_beamforming_dot, self.beamforming_update_interval)
            
    def stop_beamforming(self):
        if self.beamforming_thread is not None:
            print("TRYING TO STOP!")
            self.beamforming_thread.join()
            print("STOPPED!")
            self.beamforming_thread = None
            
        if self.beamforming_callback is not None:
            curdoc().remove_periodic_callback(self.beamforming_callback)
            self.beamforming_callback = None

    def update_beamforming(self):
        beamforming_data = self.processor.get_beamforming_results()
        self.acoustic_camera_plot.update_plot_beamforming(beamforming_data)
        
    def update_beamforming_dot(self):
        beamforming_data = self.processor.get_beamforming_results()
        self.acoustic_camera_plot.update_plot_beamforming_dots(beamforming_data)
        
        x_val = beamforming_data['max_x'][0]
        y_val = beamforming_data['max_y'][0]
        
        self.coordinates_display.text = f"X: {x_val}<br>Y: {y_val}"
        
        self.level_display.text = f"Level: {beamforming_data['max_s']}"
        
        x_deviation = x_val - self.real_x
        y_deviation = y_val - self.real_y
        z_deviation = 0
        
        if self.start_time != 0:
            elapsed_time = (datetime.datetime.now() - self.start_time).total_seconds()
        else:
            elapsed_time = 0 
        
        new_deviation_data = dict(
            time=[elapsed_time],
            x_deviation=[x_deviation],  # Durchschnittliche Abweichung
            y_deviation=[y_deviation],
            z_deviation=[z_deviation]
        )
    
        self.deviation_cds.stream(new_deviation_data, rollover=200)
    
    # Stream
    def update_stream(self):
        stream_data = self.processor.get_uma_data()
        self.stream_plot.update_plot(stream_data)

