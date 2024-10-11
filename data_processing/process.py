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


class Processor:
    def __init__(self, uma_config, mic_index, model_config_path, results_filename, ckpt_path, save_csv, save_h5, buffer_size=250):
        
        # Microphone Index of microphone-array in local system
        self.mic_index = mic_index
        
        # Configuration of the UMA
        self.uma_config = uma_config
        
        # Target frequency
        self.frequency = 2000.0
        
        self.frequency_lock = Lock()
        
        self.model_config_path = model_config_path
        self.ckpt_path = ckpt_path
        
        self.csm_buffer_size = buffer_size
        
        self.results = {
            'x': [0],
            'y': [0],
            #'z': [0],
            's': [0]
        }

        self.filename_base = results_filename
        self.save_csv = save_csv
        self.save_h5 = save_h5   
        
    def get_results(self):
        with self.result_lock:
            return self.results.copy()
        
    def start_model(self):
        print("Starting the model")
        
        # Setup the model
        self._generators_for_model()
        self._setup_model()
        
        self._model_threads()
        # Starting the thread
        print("Starting CSM thread.")
        self.csm_thread.start()
        
        print("Starting prediction thread.")
        self.compute_prediction_thread.start()


    def stop_model(self):
        print("Stopping model processing.")
        self.model_stop_event.set()
        
        self.csm_thread.join()
        print("CSM thread stopped.")
        
        self.compute_prediction_thread.join()
        print("Prediction thread stopped.")

        while not self.csm_queue.empty():
            try:
                self.csm_queue.get_nowait()
            except queue.Empty:
                break
        print("CSM queue cleared.")
         
    def _generators_for_model(self):
        print("Setting up generators for the model process.")
        # TODO: anpassen 
        # 16 channel microphone array Data Generator
        self.dev = SoundDeviceSamplesGeneratorWithPrecision(device=self.mic_index, numchannels=16)
        
        # Sample Splitter for parallel processing
        #self.sample_splitter = ac.TimeSamplesSplitter(source=self.dev)
        
        # Generator for logging the time data
        # TODO: make this work
        #self.WriteH5 = ac.WriteH5(source=self.sample_splitter, name=results_filename)
        #self.dummy_saving_generator = ac.TimeInOut(source=self.sample_splitter)

        # Real Fast Fourier Transform
        #self.fft = ac.RFFT(source=self.sample_splitter, block_size=256)
        self.fft = ac.RFFT(source=self.dev, block_size=256)
        
        # Cross Power Spectra -> CSM
        self.csm_gen = ac.CrossPowerSpectra(source=self.fft)#, norm='psd')
        
        # TODO: When MaskedSpectraInOut is implemented, use it here to filter freqencies
        # self.masked = ac.MaskedSpectraInOut(source=self.csm_gen, ...)
         
        # Index of the target frequency -> not necessary if MaskedSpectraInOut is used
        self.f_ind = np.searchsorted(self.fft.fftfreq(), self.frequency)
        
        
    def _setup_model(self):
        
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
        
        self.csm_queue = Queue(maxsize=self.csm_buffer_size)
        self.model_stop_event = Event()
        
        self.result_lock = Lock()

        # Threads und Prozesse
        self.csm_thread = Thread(target=self._csm_generator)
        self.compute_prediction_thread = Thread(target=self._predictor)
        
        
    def _csm_generator(self):
        while not self.model_stop_event.is_set():
            data = next(self.csm_gen.result(num=1))
            while not self.model_stop_event.is_set():
                try:
                    self.csm_queue.put(data, timeout=1)
                    break  # If successful, break the loop
                except queue.Full:
                    try:
                        # Oldest element is removed
                        self.csm_queue.get_nowait()
                    except queue.Empty:
                        pass  # Should not happen, but just in case
                except Exception as e:
                    print(f"Error in csm_generator: {e}")
                    break  # Unexpected error, break the loop


    def _predictor(self):
        while not self.model_stop_event.is_set():
            csm_list = []

            # Warten Sie auf das erste Element mit Timeout
            try:
                data = self.csm_queue.get(timeout=0.1)
                csm_list.append(data)
            except queue.Empty:
                # Keine Daten innerhalb des Timeouts erhalten
                continue

            # Holen Sie alle verbleibenden Elemente ohne zu blockieren
            while not self.model_stop_event.is_set():
                try:
                    data = self.csm_queue.get_nowait()
                    csm_list.append(data)
                except queue.Empty:
                    break  # Keine weiteren Daten vorhanden
                
            if len(csm_list) >= 1:
                csm_mean = np.mean(csm_list, axis=0)
                eigmode, csm_norm = self._preprocess_csm(csm_mean)
                strength_pred, loc_pred, noise_pred = self.model.predict(eigmode, verbose=0)
                strength_pred = strength_pred.squeeze()
                strength_pred *= csm_norm

                loc_pred = self.pipeline.recover_loc(loc_pred.squeeze(), aperture=self.uma_config.mics.aperture)
                
                with self.result_lock:
                    self.results = {
                        'x': loc_pred[0].tolist(),
                        'y': loc_pred[1].tolist(),
                        #'z': loc_pred[2].tolist(),
                        's': strength_pred.tolist()
                    }
                self._save_results()
                print("New Result, len of csm_list: ", len(csm_list))
                  
                
    def _preprocess_csm(self, data):
        
        # will not be necessary, as soon as MskedSpectraInOut is implemented
        csm = np.real(data).reshape(129, 16, 16)
        #csm = np.real(data).reshape(len(data)/(16*16), 16, 16)
        csm = csm[self.f_ind].reshape(self.dev.numchannels, self.dev.numchannels)

        # MEMS Empfindlichkeit anpassen
        csm = csm / 0.03548**2

        # Normalisierung zum Referenzmikrofon
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
        print("Starting beamforming.")
        
        # Setup der Generatoren für das Beamforming
        self._generators_for_beamforming()
        self._setup_beamforming()
        self._beamforming_threads()
        
        # Starten der Threads
        print("Starting beamforming thread.")
        self.beamforming_thread.start()
        print("Beamforming thread started.")
    
    def stop_beamforming(self):
        print("Stopping beamforming.")
        self.beamforming_stop_event.set()
        self.beamforming_thread.join()
        print("Beamforming thread stopped.")
        
        while not self.beamforming_queue.empty():
            try:
                self.beamforming_queue.get_nowait()
            except queue.Empty:
                break
        print("Beamforming queue cleared.")
    
    def _beamforming_threads(self):
        self.beamforming_queue = Queue(maxsize=self.csm_buffer_size)
        self.beamforming_stop_event = Event()
        self.beamforming_thread = Thread(target=self._beamforming_generator)
        self.result_lock = Lock()
    
    def _generators_for_beamforming(self):
        print("Setting up generators for beamforming.")
        # TODO: anpassen -> die Präzision ist wichtig für die Berechnung der CSM
        # 16-Kanal-Mikrofon-Array-Daten-Generator
        self.dev = ac.SoundDeviceSamplesGenerator(device=self.mic_index, numchannels=16)
        
        # Sample Splitter für parallele Verarbeitung
        self.sample_splitter = ac.TimeSamplesSplitter(source=self.dev)
        
        # Generator zum Protokollieren der Zeitdaten
        # TODO: make this work
        # self.WriteH5 = ac.WriteH5(source=self.sample_splitter, name=self.filename_base)
        self.dummy_saving_generator = ac.TimeInOut(source=self.sample_splitter)
        
        # Power Spectra für Beamforming
        self.ps = ac.PowerSpectra(source=self.sample_splitter)
        
    def _setup_beamforming(self):
        print("Setting up beamforming.")
        # TODO: Beamforming-Algorithmus initialisieren
        # Beispiel: Beamformer definieren, Grid erstellen, etc.
        # self.bb = ac.BeamformerBase(...)
        pass  # Platzhalter
    
    def _beamforming_generator(self):
        while not self.beamforming_stop_event.is_set():
            with self.frequency_lock:
                current_frequency = self.frequency
            # TODO: Beamforming-Ergebnisse berechnen
            # Beispiel für synthetische Daten:
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
            
    
    def get_beamforming_result(self):
        beamforming_list = []
        try:
            # Erstes Ergebnis mit Timeout abrufen
            data = self.beamforming_queue.get(timeout=0.1)
            beamforming_list.append(data)
        except queue.Empty:
            return None  # Keine Daten verfügbar
        
        # Weitere verfügbare Ergebnisse abrufen
        while not self.beamforming_queue.empty():
            try:
                data = self.beamforming_queue.get_nowait()
                beamforming_list.append(data)
            except queue.Empty:
                break
        
        if beamforming_list:
            # Mittelwert der Ergebnisse berechnen
            beamforming_mean = np.mean(beamforming_list, axis=0)
            return beamforming_mean
        else:
            return None
        
    def _generators_for_data(self):
        # 16 channel microphone array Data Generator
        self.dev = ac.SoundDeviceSamplesGenerator(device=self.mic_index, numchannels=16)
        
        self.recording_time = 0.1
        self.dev.numsamples = int(self.recording_time * self.dev.sample_freq)
        self.t = np.arange(self.dev.numsamples) / self.dev.sample_freq
  
    def update_frequency(self, frequency):
        with self.frequency_lock:
            self.frequency = frequency
        self.f_ind = np.searchsorted(self.fft.fftfreq(self.nfreqs), self.frequency)
        print(f"Frequency updated to {self.frequency} Hz.")
        
    def get_uma_data(self):
        signal = ac.tools.return_result(self.dev, num=256)
        return {
            'x': self.t.tolist(), 
            'y': signal.tolist()
        }
        
    def _get_current_timestamp(self):
        return datetime.datetime.now().isoformat() 
                
    def _save_results(self):
        
        timestamp = self._get_current_timestamp()
        
        current_results = self.get_results()
        
        with self.frequency_lock:
            current_frequency = self.frequency.copy()
            
        if self.save_csv:
            csv_filename = self.filename_base + '.csv'
            with open(csv_filename, mode='a', newline='') as file:
                writer = csv.writer(file)
                for x, y, s in zip(current_results['x'], current_results['y'], current_results['s']):
                    writer.writerow([timestamp, current_frequency, x, y, s])
        
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
    
