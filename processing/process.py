import numpy as np
import tensorflow as tf # type: ignore
import acoular as ac # type: ignore
import datetime
import h5py # type: ignore
import csv
from multiprocessing import Process, Event, Queue
import time
from modelsdfg.transformer.config import ConfigBase # type: ignore
from acoupipe.datasets.synthetic import DatasetSynthetic # type: ignore
from .sd_generator import SoundDeviceSamplesGeneratorWithPrecision


####################################################################################################################################
# TODOs        
# - MEMS Sensitivity 
# - Ist Processing so sinnvoll? -> overflow ist Attribut von SoundDeviceSamplesGenerator, nochmal überdenken
# - CSM Setup überdenken
# - Option auf Simples Beamforming umzuschalten hinzufügen
####################################################################################################################################


class ModelProcessor:
    def __init__(self, uma_config, mic_index, model_config_path, results_filename, ckpt_path, save_csv, save_h5, buffer_size=20, min_csm_num=20):
        # general setup
        self.mic_index = mic_index
        self.uma_config = uma_config
        self.nfreqs = 257 
        self.frequency = 5000.0
        
        # generator setup
        self.dev = SoundDeviceSamplesGeneratorWithPrecision(device=self.mic_index, numchannels=16)
        self.fft = ac.RFFT(source=self.dev) 
        self.csm_gen = ac.CrossPowerSpectra(source=self.fft, norm='psd')

        # model setup
        self.csm_shape = (self.nfreqs, self.dev.numchannels, self.dev.numchannels)
        self._setup_model(model_config_path, ckpt_path)
        
        # queue setup
        self.csm_queue = Queue(maxsize=buffer_size)
        self.stop_event = Event()
        self.min_csm_num = min_csm_num
        
        # dummy data setup
        self.noise_level = 0.05
        self.dataset_uma = DatasetSynthetic(config=self.uma_config)
        self.data_generator = self.dataset_uma.generate(f=5000, 
                                                        num=3, 
                                                        size=100, 
                                                        split='validation', 
                                                        features = ['loc'], 
                                                        progress_bar=False)
        
        # stream setup
        self.recording_time = 1
        self.dev.numsamples = int(self.recording_time * self.dev.sample_freq)
        self.t = np.arange(self.dev.numsamples) / self.dev.sample_freq
        
        # result of model prediction
        self.results = {
                'x': [0],
                'y': [0],
                's': [0]
            }
        self.filename_base = results_filename
        self.save_csv = save_csv
        self.save_h5 = save_h5
    
    # ___________________________________________________ MODEL FUNCTIONS ____________________________________________________   
    
    def _setup_model(self, model_config_path, ckpt_path):
        model_config = ConfigBase.from_toml(model_config_path)
        self.pipeline = model_config.datasets[1].pipeline.create_instance()
        self.ref_mic_index = model_config.datasets[0].pipeline.args['ref_mic_index']
        model_config.datasets[1].validation.cache = False
        self.model = tf.keras.models.load_model(ckpt_path)
        self.f_ind = np.searchsorted(self.fft.fftfreq(self.nfreqs), self.frequency)
        
    def update_frequency(self, frequency):
        self.frequency = frequency
        self.f_ind = np.searchsorted(self.fft.fftfreq(self.nfreqs), self.frequency)
        print(f"Frequency updated to {self.frequency} Hz.")
        
    def _yield_csm_to_queue(self, stop_event):
        # adds full csm to queue -> maybe this is not necessary (or sensible)
        while not stop_event.is_set():
            data = next(self.csm_gen.result(num=self.nfreqs))
            self.csm_queue.put(data)
            
    def _get_mean_csm(self):
        csm_list = []
        while len(csm_list) < self.min_csm_num and not self.stop_event.is_set():
            if not self.csm_queue.empty():
                csm_list.append(self.csm_queue.get())
        
        if len(csm_list) > 0:
            csm_mean = np.mean(csm_list, axis=0)
            return csm_mean
        else:
            return None
        
    def _preprocess_csm(self, data):
        # TODO check if this is correct
        csm = np.real(data).reshape(self.csm_shape)
        csm = csm[self.f_ind].reshape(self.dev.numchannels, self.dev.numchannels)
        # standard sensitivity of 16 mV/Pa for all microphones
        csm = csm / 0.0016**2 # TODO MEMS sensitivity
        csm_norm = csm[self.ref_mic_index, self.ref_mic_index]
        csm = csm / csm_norm
        eigmode = self.model.preprocessing(csm[np.newaxis]).numpy()
        return eigmode, csm_norm
    
    def _predict(self):
        while not self.stop_event.is_set():
            csm = self._get_mean_csm()
            
            if csm is None:
                continue
            
            eigmode, csm_norm = self._preprocess_csm(csm)
            strength_pred, loc_pred, noise_pred = self.model.predict(eigmode, verbose=0) # verbose = Terminalanzeige
            strength_pred = strength_pred.squeeze()
            strength_pred *= np.real(csm_norm)
            
            loc_pred = self.pipeline.recover_loc(loc_pred.squeeze(), aperture=self.uma_config.mics.aperture)
            
            self.results = {
                'x': loc_pred[0].tolist(),
                'y': loc_pred[1].tolist(),
                's': strength_pred.tolist()
            }
            
            self._append_results()
            
    def _beamforming(self): #TODO
        st = ac.SteeringVector(grid=self.grid, mics=self.mics, ref=[0, 0, 0])
        bb = ac.BeamformerBase(freq_data=self.csm_gen, steer=st)
        while not self.stop_event.is_set():
            pass
            
    
    def start_model(self):
        self.data_process = Process(target=self._yield_csm_to_queue, args=(self.stop_event,))
        self.data_process.start()

        try:
            self._predict()
        finally:
            self.stop_model()
            print("Model stopped.")
    
    def stop_model(self):
        while not self.csm_queue.empty():
            self.csm_queue.get()
        
        self.stop_event.set()
        self.data_process.join()
        print("Data process stopped.")
        
    # ___________________________________________________ RESULTS SAVING  ____________________________________________________
        
    def _get_current_timestamp(self):
        return datetime.datetime.now().isoformat() 
                
    def _append_results(self):
        timestamp = self._get_current_timestamp()
        
        if self.save_csv:
            csv_filename = self.filename_base + '.csv'
            with open(csv_filename, mode='a', newline='') as file:
                writer = csv.writer(file)
                for x, y, s in zip(self.results['x'], self.results['y'], self.results['s']):
                    writer.writerow([timestamp, self.frequency, x, y, s])
        
        if self.save_h5:
            h5_filename = self.filename_base + '.h5'
            with h5py.File(h5_filename, 'a') as hf:
                if 'x' not in hf:
                    hf.create_dataset('timestamp', data=np.array([timestamp]*len(self.results['x']), dtype='S19'), maxshape=(None,))
                    hf.create_dataset('frequency', data=np.array([self.frequency]*len(self.results['x'])), maxshape=(None,))
                    hf.create_dataset('x', data=np.array(self.results['x']), maxshape=(None,))
                    hf.create_dataset('y', data=np.array(self.results['y']), maxshape=(None,))
                    hf.create_dataset('s', data=np.array(self.results['s']), maxshape=(None,))  
                else:
                    for key in ['x', 'y', 's']:
                        dataset = hf[key]
                        dataset.resize((dataset.shape[0] + len(self.results[key]),))
                        dataset[-len(self.results[key]):] = self.results[key]
                    
                    timestamp_dataset = hf['timestamp']
                    timestamp_dataset.resize((timestamp_dataset.shape[0] + len(self.results['x']),))
                    timestamp_dataset[-len(self.results['x']):] = [timestamp.encode('utf-8')] * len(self.results['x'])
                    freq_dataset = hf['frequency']
                    freq_dataset.resize((freq_dataset.shape[0] + len(self.results['x']),))
                    freq_dataset[-len(self.results['x']):] = [self.frequency] * len(self.results['x'])
                    
    # ___________________________________________________ DUMMY FUNCTIONS ____________________________________________________
    
    def _dummy_predict(self):
        """Get dummy data for the UMA-16 microphone array.
        Delete this later
        """
        while not self.stop_event.is_set():
            time.sleep(0.5)
            try:
                data = next(self.data_generator)
            
            except StopIteration:
                self.data_generator = self.dataset_uma.generate(f=5000, 
                                                                num=3, 
                                                                size=100, 
                                                                split='validation', 
                                                                features=['loc'], 
                                                                progress_bar=False)
                data = next(self.data_generator)

            source_locations = data['loc'][:2].T
            strengths = np.random.rand(source_locations.shape[0]) * 20 + 60  # Random values from 60 to 80

            noisy_loc = source_locations + np.random.normal(scale=self.noise_level, size=source_locations.shape)

            self.results = {
                'x': noisy_loc[:, 0].tolist(),
                'y': noisy_loc[:, 1].tolist(),
                's': strengths.tolist()
            }

    def start_dummy_model(self):
        self.data_process = Process(target=self._yield_csm, args=(self.stop_event,))
        self.data_process.start()

        try:
            self._dummy_predict()
        finally:
            self.stop_model()
    
    # __________________________________________________ EXTRA FUNCTIONS __________________________________________________   
         
    def get_uma_data(self):
        signal = ac.tools.return_result(self.dev, num=256)
        return {
            'x': self.t.tolist(), 
            'y': signal.tolist()
        }

    