import numpy as np
import tensorflow as tf  # type: ignore
import acoular as ac  # type: ignore
from threading import Event, Lock, Thread
from queue import Queue
import queue  # Wichtig für Ausnahmen
from modelsdfg.transformer.config import ConfigBase  # type: ignore
from data_processing.sd_generator import SoundDeviceSamplesGeneratorWithPrecision
import time
from config import ConfigUMA, uma16_index 



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
        self.csm_shape = (int(self.csm_block_size/2+1), 16, 16)
        
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
    
        
    def start_model(self):
        """ Start the model processing
        """
        print("\nStarting the model.")
        
        # Call functions to setup the model
        self._generators_for_model()
        self._setup_model()
        self._model_threads()
        
        # Start thread for data logging
        print("Starting Data Saving thread.") #P1
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
        
        self.save_time_samples_thread.join() #P1
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
        sample_splitter = ac.SampleSplitter(source=self.dev, buffer_size=1024) #P1
        
        # Generator for logging the time data
        self.writeH5 = ac.WriteH5(source=sample_splitter, name=self.filename_base) #P1

        # Real Fast Fourier Transform
        self.fft = ac.RFFT(source=sample_splitter, block_size=256) #P1
        #self.fft = ac.RFFT(source=self.dev, block_size=256) #P1
        
        # Cross Power Spectra -> CSM
        self.csm_gen = ac.CrossPowerSpectra(source=self.fft)
        
        # Register the objects
        sample_splitter.register_object(self.fft, self.writeH5) #P1
        
        # TODO: When MaskedSpectraInOut is implemented, use it here to filter freqencies
        # Problem: This means, we have to restart the model setup, when the frequency is updated
        #          Maybe this is a good thing?
        # Filter the frequencies
        # self.masked = ac.MaskedSpectraOut(source=self.csm_gen, ...) #P2

        # Index of the target frequency -> not necessary if MaskedSpectraInOut is used
        self.f_ind = np.searchsorted(self.fft.fftfreq(), self.frequency) #P2
        
    # TODO: make this work   
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
        self.save_time_samples_thread = Thread(target=self._save_time_samples) #P1
        
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


config_uma = ConfigUMA()
mic_index = uma16_index()
model_dir = "/home/rabea/Documents/Bachelorarbeit/models/EigmodeTransformer_learning_rate0.00025_epochs100_2024-10-09_09-03"
model_config_path = model_dir + "/config.toml"
ckpt_path = model_dir + '/ckpt/best_ckpt/0078-1.06.keras'
results_filename = "test_results"
 
ac.config.global_caching = 'none' # type: ignore

processor = Processor(
    config_uma,
    mic_index,
    model_config_path,
    results_filename,
    ckpt_path,
    save_csv=False,
    save_h5=False
)

processor.start_model()
time.sleep(5)
processor.stop_model()
