import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import numpy as np
import tensorflow as tf  # type: ignore
import acoular as ac  # type: ignore
import datetime
import h5py  # type: ignore
import csv
from threading import Event, Lock, Thread
from pathlib import Path
from queue import Queue
import time
import queue  # Wichtig für Ausnahmen
from .SamplesProcessor import LastInOut
from .sd_generator import SoundDeviceSamplesGeneratorWithPrecision


class Processor:
    def __init__(self, device, results_folder, ckpt_path, save_csv, save_h5, 
                 csm_block_size=256, csm_min_queue_size=60, csm_buffer_size=250):
        """ Processor for the UMA16 Acoustic Camera
        """
        # Microphone Index of microphone-array in local system
        self.device = device
        micgeom_path = Path(ac.__file__).parent / 'xml' / 'minidsp_uma-16_mirrored.xml'
        self.mics = ac.MicGeom(from_file=micgeom_path)
        
        x_min, x_max, y_min, y_max = -1.5, 1.5, -1.5, 1.5
        z = 1.25
        increment = 0.1
        self.beamforming_grid = ac.RectGrid(x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, z=z, increment=increment)
        self.grid_dim = (int((x_max - x_min) / increment + 1), int((y_max - y_min) / increment + 1))
        
        # Default target frequency
        self.frequency = 4000.0
        
        # Frequencies for the third octave bands
        self.third_octave_frequencies = [250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000]
        
        # Locks for adjustable parameters
        self.frequency_lock = Lock()
        self.csm_block_size_lock = Lock()
        self.min_queue_size_lock = Lock()
        self.z = Lock()
        
        # Lock for the results
        self.result_lock = Lock()
        
        # Lock for beamforming results
        self.beamforming_result_lock = Lock()
        
        # Path to the model checkpoint
        self.ckpt_path = ckpt_path
        
        # Size of the buffer in CSM queue
        self.csm_buffer_size = csm_buffer_size
        
        # Size of one block in CSM
        self.csm_block_size = csm_block_size
        
        # Minimum size of the CSM queue
        self.min_queue_size = csm_min_queue_size
        
        self.csm_num = 1
        
        # Shape of the CSM
        self.csm_shape = (int(csm_block_size/2+1), 16, 16)
        
        # Results dictionary that will be updated
        self.results = {
            'x': [0],
            'y': [0],
            'z': [0],
            's': [0]
        }
        
        base_beamforming_result = np.zeros(self.grid_dim).tolist()
        
        self.beamforming_results = {'results' : base_beamforming_result}
        
        # Filename for the results
        self.results_folder = results_folder
        
        # Boolean flags for saving the results
        self.save_csv = save_csv
        self.save_h5 = save_h5   
        
        # For testing purposes only: data of the first channel can be plotted
        self._generators_for_data() 
        self.data_filename, self.results_filename = self._get_result_filenames('model')
        
        # Call functions to setup the model
        self._generators_for_model()
        
        
    def start_model(self):
        """ Start the model processing
        """
        print("\nStarting the model.")
        
        # filenames
        self.data_filename, self.results_filename = self._get_result_filenames('model')
        
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
        self.dev = SoundDeviceSamplesGeneratorWithPrecision(device=self.device, numchannels=16) #P0
        
        # Sample Splitter for parallel processing
        sample_splitter = ac.SampleSplitter(source=self.dev, buffer_size=1024) 
        
        # Generator for logging the time data
        self.writeH5 = ac.WriteH5(source=sample_splitter, name=f"{self.data_filename}.h5") 

        # Real Fast Fourier Transform
        self.fft = ac.RFFT(source=sample_splitter, block_size=self.csm_block_size)
        
        # Cross Power Spectra -> CSM
        self.csm_gen = ac.CrossPowerSpectra(source=self.fft)
        
        # Register the objects
        sample_splitter.register_object(self.fft, self.writeH5)
        
        # TODO: When MaskedSpectraOut is implemented, use it here to filter freqencies
        # Problem: This means, we have to restart the model setup, when the frequency is updated
        #          Maybe this is a good thing?
        # Filter the frequencies
        # self.masked = ac.MaskedSpectraOut(source=self.csm_gen, ...)

        # Index of the target frequency -> not necessary if MaskedSpectraOut is used
        self.f_ind = np.searchsorted(self.fft.fftfreq(), self.frequency)
        
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

        self.ref_mic_index = 0
        
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
            data = next(self.csm_gen.result(num=self.csm_num))
            
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

            if self.csm_queue.qsize() < self.min_queue_size:
                time.sleep(0.1)
                continue   
                
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
                
            # Calculate the mean of the CSM data
            csm_mean = np.mean(csm_list, axis=0)
            #np.save("csm.npy", csm_mean)
            
            # Preprocess the CSM data
            eigmode, csm_norm = self._preprocess_csm(csm_mean)
            #np.save("eigmode.npy", eigmode)
            
            # Predict the strength and location
            strength_pred, loc_pred, noise_pred = self.model.predict(eigmode, verbose=0)
            strength_pred = strength_pred.squeeze()
            strength_pred *= csm_norm
            #strength_pred *= np.real(csm_norm)[:,np.newaxis]
            
            # TODO Adam ändert das noch
            #loc_pred = self._recover_loc(loc_pred.squeeze(), aperture=self.mics.aperture) 
            loc_pred -= 0.5
            loc_pred *= 2.0
            loc_pred = loc_pred.squeeze()
            loc_pred = loc_pred.squeeze()
            # Update the results
            with self.result_lock:
                self.results = {
                    'x': loc_pred[0].tolist(),
                    'y': loc_pred[1].tolist(),
                    'z': loc_pred[2].tolist(),
                    's': strength_pred.tolist()
                }

            self._save_results()
    
    def _recover_loc(self, loc, aperture, shift_loc=True, norm_loc=2):
        if shift_loc:
            if isinstance(shift_loc, float):
                loc = loc - shift_loc
            else:
                loc = loc - 0.5
        if norm_loc:
            if isinstance(norm_loc, float):
                loc = loc * norm_loc
            else:
                loc = loc * aperture
        return loc
                
    def _preprocess_csm(self, data):
        """ Preprocess the CSM data
        """
        # will not be necessary, as soon as MsskedSpectraInOut is implemented
        csm = np.real(data).reshape(self.csm_shape)
        #csm = np.real(data).reshape(len(data)/(16*16), 16, 16)
        csm = csm[self.f_ind].reshape(self.dev.numchannels, self.dev.numchannels)
        
        # at this point, the CSM is in [Volt^2] since the microphone channel data
        # is a voltage strea -> we need to convert it to squared sound pressure values [Pa^2]
        # we do not account for the true sensitivity of the MEMS microphones here! 
        # I just use a standard sensitivity of 16 mV/Pa for all microphones
        # MEMS Sensitivity (94 dB SPL @ 1 kHz) -29 dBFS -> TODO?
        csm = csm / 0.0016**2

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
    
    def _preprocess_csms_multiple_freqs(self, data):
        csm_bands = []
        for f_ind in self.freq_indices:
            csm_band = np.real(data[f_ind]).reshape(self.dev.numchannels, self.dev.numchannels)
            csm_bands.append(csm_band)
        
        csm_norms = [band[self.ref_mic_index, self.ref_mic_index] for band in csm_bands]
        csm_bands = [band / norm for band, norm in zip(csm_bands, csm_norms)]
        
        eigmodes = []
        for band in csm_bands:
            evls, evecs = np.linalg.eigh(band)
            eigmode = evecs[..., -8:] * evls[:, np.newaxis, -8:]
            eigmode = np.stack([np.real(eigmode), np.imag(eigmode)], axis=3)
            eigmode = np.transpose(eigmode, [0, 2, 1, 3])
            eigmodes.append(np.reshape(eigmode, [-1, eigmode.shape[1], eigmode.shape[2]*eigmode.shape[3]]))

        return eigmodes, csm_norms
    
    def _get_freq_indicies(self, freqs):
        freq_indices = [np.searchsorted(self.fft.fftfreq(), freq) for freq in freqs]
        return freq_indices

    def start_beamforming(self):
        """ Start the beamforming process
        """
        print("\nStarting beamforming.")
        
        # filenames
        self.data_filename, self.results_filename = self._get_result_filenames('beamforming')
        
        # Setup the generators for the beamforming process
        self._generators_for_beamforming()
        self._beamforming_threads()
        
        # Start the beamforming thread
        self.beamforming_thread.start()   
        print("Beamforming thread started.")
        
        self.save_time_samples_beamforming_thread.start()
        print("Time data saving thread started.")
    
    def stop_beamforming(self):
        """ Stop the beamforming process
        """
        print("Stopping beamforming.")
        
        # Set the event to stop all threads
        self.beamforming_stop_event.set()
        # End the beamforming thread
        self.beamforming_thread.join()
        print("Beamforming thread stopped.")
        
        self.save_time_samples_beamforming_thread.join()
        print("Time data saving thread stopped.")
        
    def get_beamforming_results(self):
        """ Get current results of the model
        """
        # Return a copy of the results safely
        with self.beamforming_result_lock:
            return self.beamforming_results.copy()
    
    def _generators_for_beamforming(self):
        """ Setup the generators for the beamforming process
        """
        print("Setting up generators for beamforming.")
        
        # 16-Kanal-Mikrofon-Array-Daten-Generator
        self.dev = ac.SoundDeviceSamplesGenerator(device=self.device, numchannels=16)
        
        # Turn Volt to Pascal 
        self.source_mixer = ac.SourceMixer(sources=[self.dev],weights=np.array([1/0.0016]))
        
        # Sample Splitter
        sample_splitter = ac.SampleSplitter(source=self.source_mixer, buffer_size=1024)
        
        # Generator for Logging time data
        self.writeH5 = ac.WriteH5(source=sample_splitter, name=self.data_filename)

        # Steering Vector
        steer = ac.SteeringVector(env=ac.Environment(c=343), grid=self.beamforming_grid, mics=self.mics)
        
        self.lastOut = LastInOut(source=sample_splitter)
        self.bf = ac.BeamformerTime(source=self.lastOut, steer=steer)
        
        filter = ac.FiltOctave(source=self.bf, band=self.frequency, fraction='Third octave')
        power = ac.TimePower(source=filter)
        self.bf_out = ac.TimeAverage(source=power, naverage=512)
        
        sample_splitter.register_object(self.lastOut, self.writeH5)
        
    def _beamforming_threads(self):
        """ Threads for the beamforming process
        """
        # Event to eventually stop all threads
        self.beamforming_stop_event = Event()
        
        # Thread
        self.beamforming_thread = Thread(target=self._beamforming_generator)
        self.save_time_samples_beamforming_thread = Thread(target=self._save_time_samples_beamforming)
    
    def _beamforming_generator(self):
        """ Beamforming-Generator """
        gen = self.bf_out.result(num=1)
        count = 0
        
        while not self.beamforming_stop_event.is_set():
            try:
                res = ac.L_p(next(gen))
                res = res.reshape(self.grid_dim)
                count += 1
                with self.beamforming_result_lock:
                    self.beamforming_results['results'] = res.tolist()  
                
            except StopIteration:
                print("Generator has been stopped.")
                break
            
            except Exception as e:
                print(f"Exception in _beamforming_generator: {e}")
                break
        print(f"Beamforming: Calculated {count} results.")
              
    def _save_time_samples_beamforming(self):
        """ Save the time samples to a H5 file """
        gen = self.writeH5.result(num=512) 
        block_count = 0
        
        while not self.beamforming_stop_event.is_set():
            try:
                next(gen)
                block_count += 1
                
            except StopIteration:
                break
            
        print("Finished saving time samples.")
        print(f"Saved {block_count} blocks.")
              
    def update_frequency(self, frequency):
        """ Update the target frequency for the model
        """
        with self.frequency_lock:
            self.frequency = frequency
        #self.f_ind = np.searchsorted(self.fft.fftfreq(), self.frequency)
        print(f"Frequency updated to {self.frequency} Hz.")
        self.f_ind = self.fft.fftfreq()-  self.frequency
        
    def update_csm_block_size(self, block_size):
        """ Update the block size for the CSM
        """
        with self.csm_block_size_lock:
            self.csm_block_size = block_size
            self.csm_shape = (int(block_size/2+1), 16, 16)
        print(f"CSM block size updated to {self.csm_block_size}.")
        
    def update_min_queue_size(self, min_queue_size):
        """ Update the minimum queue size for the CSM
        """
        with self.min_queue_size_lock:
            self.min_queue_size = min_queue_size
        print(f"Minimum queue size updated to {self.min_queue_size}.")   
    
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
        self.dev_data = ac.SoundDeviceSamplesGenerator(device=self.device, numchannels=16)
        
        # Length of the recording
        self.recording_time = 0.1
        self.dev_data.numsamples = int(self.recording_time * self.dev_data.sample_freq)
        self.t = np.arange(self.dev_data.numsamples) / self.dev_data.sample_freq
        
    def _get_current_timestamp(self):
        """ Get the current timestamp in ISO format
        """
        return datetime.datetime.now().isoformat() 
    
    def _get_result_filenames(self, type):
        """ Get the filenames for the results
        """
        current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        
        data_filename = self.results_folder + f'/{current_time}_{type}_time_data'
        result_filename = self.results_folder + f'/{current_time}_{type}_results' 
        
        return data_filename, result_filename
                
    def _save_results(self):
        """ Save the results to a CSV and H5 file
        """
        timestamp = self._get_current_timestamp()
        current_results = self.get_results()
        
        with self.frequency_lock:
            current_frequency = self.frequency
        
        # Save the results to a CSV file
        if self.save_csv:
            csv_filename = self.results_filename + '.csv'
            with open(csv_filename, mode='a', newline='') as file:
                writer = csv.writer(file)
                for x, y, s in zip(current_results['x'], current_results['y'], current_results['s']):
                    writer.writerow([timestamp, current_frequency, x, y, s])
        
        # Save the results to a H5 file
        if self.save_h5:
            h5_filename = self.results_filename + '.h5'
            with h5py.File(h5_filename, 'a') as hf:
                if 'x' not in hf:
                    hf.create_dataset('timestamp', data=np.array([timestamp]*len(current_results['x']), dtype='S19'), maxshape=(None,))
                    hf.create_dataset('frequency', data=np.array([current_frequency]*len(current_results['x'])), maxshape=(None,))
                    hf.create_dataset('x', data=np.array(current_results['x']), maxshape=(None,))
                    hf.create_dataset('y', data=np.array(current_results['y']), maxshape=(None,))
                    hf.create_dataset('z', data=np.array(current_results['z']), maxshape=(None,))
                    hf.create_dataset('s', data=np.array(current_results['s']), maxshape=(None,))  
                else:
                    for key in ['x', 'y', 'z', 's']:
                        dataset = hf[key]
                        dataset.resize((dataset.shape[0] + len(current_results[key]),))
                        dataset[-len(current_results[key]):] = current_results[key]
                    
                    timestamp_dataset = hf['timestamp']
                    timestamp_dataset.resize((timestamp_dataset.shape[0] + len(current_results['x']),))
                    timestamp_dataset[-len(current_results['x']):] = [timestamp.encode('utf-8')] * len(current_results['x'])
                    freq_dataset = hf['frequency']
                    freq_dataset.resize((freq_dataset.shape[0] + len(current_results['x']),))
                    freq_dataset[-len(current_results['x']):] = [current_frequency] * len(current_results['x'])
    
