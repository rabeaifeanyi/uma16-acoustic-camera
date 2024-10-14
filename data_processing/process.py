import numpy as np
import tensorflow as tf  # type: ignore
import acoular as ac  # type: ignore
import datetime
import h5py  # type: ignore
import csv
from threading import Event, Lock, Thread
from queue import Queue
import queue  # Wichtig für Ausnahmen
from modelsdfg.transformer.config import ConfigBase  # type: ignore
from .sd_generator import SoundDeviceSamplesGeneratorWithPrecision

# Problems
# 2: MaskedSpectraInOut not yet implemented #P2
# 3: Beamforming not implemented #P3
# 4: Saving of results not tested #P4
# 0: All sorts of small problems #P0

class Processor:
    def __init__(self, uma_config, mic_index, model_config_path, results_filename, ckpt_path, save_csv, save_h5, 
                 csm_buffer_size=250, beamforming_buffer_size=150, csm_block_size=256):
        """ Processor for the UMA16 Acoustic Camera

        """
        # Microphone Index of microphone-array in local system
        self.mic_index = mic_index
        
        # Configuration of the UMA
        self.uma_config = uma_config
        
        # Default target frequency
        self.frequency = 2000.0
        
        # Lock for the frequency
        self.frequency_lock = Lock()
        
        # Lock for the results
        self.result_lock = Lock()
        
        # Path to the model configuration folder
        self.model_config_path = model_config_path
        
        # Path to the model checkpoint
        self.ckpt_path = ckpt_path
        
        # Size of the buffer in CSM queue
        self.csm_buffer_size = csm_buffer_size
        
        # Size of one block in CSM
        self.csm_block_size = csm_block_size
        
        # Shape of the CSM
        self.csm_shape = (int(csm_block_size/2+1), 16, 16)
        
        # Size of the buffer in beamforming queue
        self.beamforming_buffer_size = beamforming_buffer_size
        
        # Results dictionary that will be updated
        self.results = {
            'x': [0],
            'y': [0],
            'z': [0],
            's': [0]
        }

        # Filename for the results
        self.filename_base = results_filename
        
        # Boolean flags for saving the results
        self.save_csv = save_csv
        self.save_h5 = save_h5   
        
        # For testing purposes only: data of the first channel can be plotted
        self._generators_for_data() 
        
    def start_model(self):
        """ Start the model processing
        """
        print("\nStarting the model.")
        
        # Call functions to setup the model
        self._generators_for_model()
        self._setup_model()
        self._model_threads()
        
        # Start thread for data logging
        print("Starting Data Saving thread.")
        self.save_time_samples_thread.start()
        
        # Start thread for CSM generation
        print("Starting CSM thread.")
        self.csm_thread.start()
        
        # Start thread for prediction
        print("Starting prediction thread.")
        self.compute_prediction_thread.start()

    def stop_model(self):
        """ Stop the model processing
        """
        print("Stopping model processing.")
        
        # Set Event to stop all inner threads
        self.model_stop_event.set()
        
        # End all threads
        self.csm_thread.join()
        print("CSM thread stopped.")
        
        self.compute_prediction_thread.join() 
        print("Prediction thread stopped.")
        
        self.save_time_samples_thread.join()
        print("Data Saving thread stopped.")

        # Clear the CSM queue
        while not self.csm_queue.empty():
            try:
                self.csm_queue.get_nowait()
            except queue.Empty:
                break
        print("CSM queue cleared.")
        
    def get_results(self):
        """ Get current results of the model
        """
        # Return a copy of the results safely
        with self.result_lock:
            return self.results.copy()
         
    def _generators_for_model(self):
        """ Setup the generators for the model process
        """
        print("Setting up generators for the model process.")
        # TODO: anpassen, falls möglich 
        # Data Generator
        self.dev = SoundDeviceSamplesGeneratorWithPrecision(device=self.mic_index, numchannels=16) #P0
        
        # Sample Splitter for parallel processing
        sample_splitter = ac.SampleSplitter(source=self.dev, buffer_size=1024) 
        
        # Generator for logging the time data
        self.writeH5 = ac.WriteH5(source=sample_splitter, name=f"{self.filename_base}.h5") 

        # Real Fast Fourier Transform
        self.fft = ac.RFFT(source=sample_splitter, block_size=self.csm_block_size)
        
        # Cross Power Spectra -> CSM
        self.csm_gen = ac.CrossPowerSpectra(source=self.fft)
        
        # Register the objects
        sample_splitter.register_object(self.fft, self.writeH5)
        
        # TODO: When MaskedSpectraInOut is implemented, use it here to filter freqencies
        # Problem: This means, we have to restart the model setup, when the frequency is updated
        #          Maybe this is a good thing?
        # Filter the frequencies
        # self.masked = ac.MaskedSpectraInOut(source=self.csm_gen, ...) #P2

        # Index of the target frequency -> not necessary if MaskedSpectraInOut is used
        self.f_ind = np.searchsorted(self.fft.fftfreq(), self.frequency) #P2
        
    def _save_time_samples(self):
        """ Save the time samples to a H5 file """
        gen = self.writeH5.result(num=self.csm_block_size)
        block_count = 0
        
        while not self.model_stop_event.is_set():
            try:
                next(gen)
                block_count += 1
                
            except StopIteration:
                break
            
        print("Finished saving time samples.")
        print(f"Saved {block_count} blocks.")
        
    def _setup_model(self):
        """ Setup the model for the prediction
        """
        print("Setting up model.")
        
        # Load the model configuration
        model_config = ConfigBase.from_toml(self.model_config_path)
        
        # Load the pipeline for the model
        self.pipeline = model_config.datasets[1].pipeline.create_instance()
        
        # Load the validation pipeline
        self.ref_mic_index = model_config.datasets[0].pipeline.args['ref_mic_index']

        model_config.datasets[1].validation.cache = False
        
        # Load the model
        self.model = tf.keras.models.load_model(self.ckpt_path)
        
        
    def _model_threads(self):
        """ Threads for the model process
        """
        # Queue for the CSM data
        self.csm_queue = Queue(maxsize=self.csm_buffer_size)
        
        # Event to eventually stop all threads
        self.model_stop_event = Event()

        # Threads
        self.csm_thread = Thread(target=self._csm_generator)
        self.compute_prediction_thread = Thread(target=self._predictor)
        self.save_time_samples_thread = Thread(target=self._save_time_samples) 
        
    def _csm_generator(self):
        """ CSM generator thread for the model
        """
        while not self.model_stop_event.is_set():
            
            # Yield the next CSM data block
            data = next(self.csm_gen.result(num=1))
            
            while not self.model_stop_event.is_set():
                try:
                    self.csm_queue.put(data, timeout=1)
                    break  # If successful, break the loop
                
                # Queue is full, remove the oldest element
                except queue.Full:
                    try:  
                        # Oldest element is removed
                        self.csm_queue.get_nowait()
                    
                    # Should not happen, but just in case
                    except queue.Empty:
                        pass
                
                # Unexpected error, break the loop    
                except Exception as e:
                    print(f"Error in csm_generator: {e}")
                    break  

    def _predictor(self):
        """ Prediction thread for the model
        """
        while not self.model_stop_event.is_set():
            csm_list = []

            # Wait for the next CSM data block
            try:
                data = self.csm_queue.get(timeout=0.1)
                csm_list.append(data)
            except queue.Empty:
                # Timeout, no data available
                continue

            # Collect all available data
            while not self.model_stop_event.is_set():
                try:
                    data = self.csm_queue.get_nowait()
                    csm_list.append(data)
                except queue.Empty:
                    break  # No more data available
            
            # If data is available, process it    
            if len(csm_list) >= 1:
                
                # Calculate the mean of the CSM data
                csm_mean = np.mean(csm_list, axis=0)
                
                # Preprocess the CSM data
                eigmode, csm_norm = self._preprocess_csm(csm_mean)
                
                # Predict the strength and location
                strength_pred, loc_pred, noise_pred = self.model.predict(eigmode, verbose=0)
                strength_pred = strength_pred.squeeze()
                strength_pred *= csm_norm

                # Recover the location
                loc_pred = self.pipeline.recover_loc(loc_pred.squeeze(), aperture=self.uma_config.mics.aperture)
                
                # Update the results
                with self.result_lock:
                    self.results = {
                        'x': loc_pred[0].tolist(),
                        'y': loc_pred[1].tolist(),
                        'z': loc_pred[2].tolist(),
                        's': strength_pred.tolist()
                    }
                    
                #self._save_results() #P4
                
    def _preprocess_csm(self, data):
        """ Preprocess the CSM data
        """
        # will not be necessary, as soon as MsskedSpectraInOut is implemented
        csm = np.real(data).reshape(self.csm_shape) #P2
        #csm = np.real(data).reshape(len(data)/(16*16), 16, 16)
        csm = csm[self.f_ind].reshape(self.dev.numchannels, self.dev.numchannels) #P2

        # MEMS Sensitivity
        csm = csm / 0.03548**2 #P0

        # Normalization of the CSM with respect to the reference microphone
        csm_norm = csm[self.ref_mic_index, self.ref_mic_index]
        csm = csm / csm_norm
        csm = csm.reshape(1, 16, 16)

        # Preprocessing
        neig = 8
        evls, evecs = np.linalg.eigh(csm)
        eigmode = evecs[..., -neig:] * evls[:, np.newaxis, -neig:]
        eigmode = np.stack([np.real(eigmode), np.imag(eigmode)], axis=3)
        eigmode = np.transpose(eigmode, [0, 2, 1, 3])
        input_shape = np.shape(eigmode)
        eigmode = np.reshape(eigmode, [-1, input_shape[1], input_shape[2]*input_shape[3]])

        return eigmode, csm_norm

    def start_beamforming(self):
        """ Start the beamforming process
        """
        print("\nStarting beamforming.")
        
        # Setup the generators for the beamforming process
        self._generators_for_beamforming()
        self._setup_beamforming()
        self._beamforming_threads()
        
        # Start the beamforming thread
        self.beamforming_thread.start()   
        print("Beamforming thread started.")
    
    def stop_beamforming(self):
        """ Stop the beamforming process
        """
        print("Stopping beamforming.")
        
        # Set the event to stop all threads
        self.beamforming_stop_event.set()
        
        # End the beamforming thread
        self.beamforming_thread.join()
        print("Beamforming thread stopped.")
        
        # Clear the beamforming queue
        while not self.beamforming_queue.empty():
            try:
                self.beamforming_queue.get_nowait()
            except queue.Empty:
                break
        print("Beamforming queue cleared.")
    
    def _generators_for_beamforming(self):
        """ Setup the generators for the beamforming process
        """
        print("Setting up generators for beamforming.")
        # TODO: anpassen -> die Präzision ist wichtig für die Berechnung der CSM
        # 16-Kanal-Mikrofon-Array-Daten-Generator
        self.dev = ac.SoundDeviceSamplesGenerator(device=self.mic_index, numchannels=16)
        
        # Sample Splitter für parallele Verarbeitung
        #self.sample_splitter = ac.TimeSamplesSplitter(source=self.dev)
        
        # Generator zum Protokollieren der Zeitdaten
        # TODO: make this work
        # self.WriteH5 = ac.WriteH5(source=self.sample_splitter, name=self.filename_base)
        #self.dummy_saving_generator = ac.TimeInOut(source=self.sample_splitter)

        # TODO replace with time beamforming
        self.ps = ac.PowerSpectra(source=self.dev)
        
    def _setup_beamforming(self):
        """ Setup the beamforming algorithm
        """
        print("Setting up beamforming.")
        # TODO: Beamforming-Algorithmus initialisieren
        # self.bb = ac.BeamformerBase(...)
        pass  # Platzhalter
    
    def _beamforming_threads(self):
        """ Threads for the beamforming process
        """
        # Queue for the beamforming data
        self.beamforming_queue = Queue(maxsize=self.beamforming_buffer_size)
        
        # Event to eventually stop all threads
        self.beamforming_stop_event = Event()
        
        # Thread
        self.beamforming_thread = Thread(target=self._beamforming_generator)
    
    def _beamforming_generator(self):
        """ Beamforming-Generator
        """
        while not self.beamforming_stop_event.is_set():
            with self.frequency_lock:
                current_frequency = self.frequency
            # TODO: Beamforming-Ergebnisse berechnen
            # pm = self.bb.synthetic(self.frequency, 3)
            # Lm = ac.L_p(pm)
            # self.beamforming_queue.put(Lm)
            
            # Platzhalter für echte Beamforming-Berechnung:
            # pm = self.bb.synthetic(self.frequency, 3)
            # Lm = ac.L_p(pm)
            # self.beamforming_queue.put(Lm)
            
            # Für das Beispiel verwenden wir Dummy-Daten:
            dummy = np.random.rand(64, 64)
            self.beamforming_queue.put(dummy)
              
    def update_frequency(self, frequency):
        """ Update the target frequency for the model
        """
        with self.frequency_lock:
            self.frequency = frequency
        self.f_ind = np.searchsorted(self.fft.fftfreq(), self.frequency)
        print(f"Frequency updated to {self.frequency} Hz.")
    
    def get_uma_data(self):
        """  Returns the time data of the microphone array      
        """
        signal = ac.tools.return_result(self.dev_data, num=256)
        return {
            'x': self.t.tolist(), 
            'y': signal.tolist()
        }
        
    def _generators_for_data(self):
        """ Setup the generators for the data processing
        """
        # 16-Channel-Microphone-Array-Data-Generator
        self.dev_data = ac.SoundDeviceSamplesGenerator(device=self.mic_index, numchannels=16)
        
        # Length of the recording
        self.recording_time = 0.1
        self.dev_data.numsamples = int(self.recording_time * self.dev_data.sample_freq)
        self.t = np.arange(self.dev_data.numsamples) / self.dev_data.sample_freq
        
    def _get_current_timestamp(self):
        """ Get the current timestamp in ISO format
        """
        return datetime.datetime.now().isoformat() 
                
    def _save_results(self): #P4
        """ Save the results to a CSV and H5 file
        """
        timestamp = self._get_current_timestamp()
        
        current_results = self.get_results()
        
        with self.frequency_lock:
            current_frequency = self.frequency
        
        # Save the results to a CSV file
        if self.save_csv:
            csv_filename = self.filename_base + '.csv'
            with open(csv_filename, mode='a', newline='') as file:
                writer = csv.writer(file)
                for x, y, s in zip(current_results['x'], current_results['y'], current_results['s']):
                    writer.writerow([timestamp, current_frequency, x, y, s])
        
        # Save the results to a H5 file
        if self.save_h5:
            h5_filename = self.filename_base + '.h5'
            with h5py.File(h5_filename, 'a') as hf:
                if 'x' not in hf:
                    hf.create_dataset('timestamp', data=np.array([timestamp]*len(current_results['x']), dtype='S19'), maxshape=(None,))
                    hf.create_dataset('frequency', data=np.array([current_frequency]*len(current_results['x'])), maxshape=(None,))
                    hf.create_dataset('x', data=np.array(current_results['x']), maxshape=(None,))
                    hf.create_dataset('y', data=np.array(current_results['y']), maxshape=(None,))
                    hf.create_dataset('s', data=np.array(current_results['s']), maxshape=(None,))  
                else:
                    for key in ['x', 'y', 's']:
                        dataset = hf[key]
                        dataset.resize((dataset.shape[0] + len(current_results[key]),))
                        dataset[-len(current_results[key]):] = current_results[key]
                    
                    timestamp_dataset = hf['timestamp']
                    timestamp_dataset.resize((timestamp_dataset.shape[0] + len(current_results['x']),))
                    timestamp_dataset[-len(current_results['x']):] = [timestamp.encode('utf-8')] * len(current_results['x'])
                    freq_dataset = hf['frequency']
                    freq_dataset.resize((freq_dataset.shape[0] + len(current_results['x']),))
                    freq_dataset[-len(current_results['x']):] = [current_frequency] * len(current_results['x'])
    