import numpy as np
import tensorflow as tf # type: ignore
import acoular as ac # type: ignore
import datetime
import h5py # type: ignore
import time
import csv
from multiprocessing import Process, Event, Queue
import multiprocessing
from modelsdfg.transformer.config import ConfigBase # type: ignore
from .sd_generator import SoundDeviceSamplesGeneratorWithPrecision


####################################################################################################################################
# TODOs        
# - Zeitsignale loggen -> Sample Splitter oder "nachschalten"
####################################################################################################################################

class Processor:
    def __init__(self, uma_config, mic_index, model_config_path, results_filename, ckpt_path, save_csv, save_h5, buffer_size=20, min_csm_num=20):

        self.mic_index = mic_index
        self.uma_config = uma_config
        self.nfreqs = 257 
        self.frequency = 2000.0
        
        self.dev = SoundDeviceSamplesGeneratorWithPrecision(device=self.mic_index, numchannels=16)

        self.fft = ac.RFFT(source=self.dev) 
        self.csm_gen = ac.CrossPowerSpectra(source=self.fft, norm='psd')
        self.csm_shape = (self.nfreqs, self.dev.numchannels, self.dev.numchannels)
        
        self.f_ind = np.searchsorted(self.fft.fftfreq(self.nfreqs), self.frequency) #TODO
        self._setup_model(model_config_path, ckpt_path)
        
        self.ps = ac.PowerSpectra(source=self.dev)
        self._setup_beamforming()        
        
        self.csm_queue = Queue(maxsize=buffer_size)
        self.beamforming_queue = Queue(maxsize=buffer_size)
        
        self.model_stop_event = Event()
        self.beamforming_stop_event = Event()
        
        self.min_csm_num = min_csm_num
        
        self.recording_time = 0.1
        self.dev.numsamples = int(self.recording_time * self.dev.sample_freq)
        self.t = np.arange(self.dev.numsamples) / self.dev.sample_freq
        
        self.results = {
                'x': [0],
                'y': [0],
                's': [0]
            }
        
        self.beamforming_results = {
                's': [0]
            }
        
        self.filename_base = results_filename
        self.save_csv = save_csv
        self.save_h5 = save_h5
    
    def _setup_model(self, model_config_path, ckpt_path):
        print("Setting up model.")
        model_config = ConfigBase.from_toml(model_config_path)
        self.pipeline = model_config.datasets[1].pipeline.create_instance()
        self.ref_mic_index = model_config.datasets[0].pipeline.args['ref_mic_index']
        model_config.datasets[1].validation.cache = False
        self.model = tf.keras.models.load_model(ckpt_path)
        
    def _setup_beamforming(self):
        print("Setting up beamforming.")
        self.mg = self.uma_config.create_mics()
        self.ps = ac.PowerSpectra(time_data=self.dev, block_size=128, window='Hanning')
        self.rg = self.uma_config.create_grid()
        self.st = ac.SteeringVector(grid=self.rg, mics=self.mg)
        self.bb = ac.BeamformerBase(freq_data=self.ps, steer=self.st)
        self.f_ind_beamforming = np.searchsorted(self.ps.fftfreq(), self.frequency)
        
    def _yield_csm_to_queue(self, stop_event):
        while not stop_event.is_set():
            data = next(self.csm_gen.result(num=self.nfreqs))
            self.csm_queue.put(data)
            
    def _get_mean_csm(self):
        csm_list = []
        while len(csm_list) < self.min_csm_num and not self.model_stop_event.is_set():
            if not self.csm_queue.empty():
                csm_list.append(self.csm_queue.get())
        
        if len(csm_list) > 0:
            csm_mean = np.mean(csm_list, axis=0)
            return csm_mean
        else:
            return None
        
    def _preprocess_csm(self, data):
        csm = np.real(data).reshape(self.csm_shape)

        # pick out one frequency
        csm = csm[self.f_ind].reshape(self.dev.numchannels, self.dev.numchannels)
        
        # MEMS sensitivity: -29 dBFS -> 1Pa Output is 29 dB 
        csm = csm / 0.03548**2
        
        # Normalise to reference mic
        csm_norm = csm[self.ref_mic_index, self.ref_mic_index]
        csm = csm / csm_norm
        
        # Preprocessing:
        #evls, evecs = tf.linalg.eigh(csm)
        #evecs[...,-neig:]*evls[:,tf.newaxis,-neig:]
        eigmode = self.model.preprocessing(csm[np.newaxis]).numpy()
        return eigmode, csm_norm
    
    def _predict(self):
        #TODO unbedingt nachfragen: Was passiert hier, wie wurde model trainiert etc. 
        # warum wird immer (2,8) vorhergesagt?
        while not self.model_stop_event.is_set():
            csm = self._get_mean_csm()
            
            if csm is None:
                continue
            
            eigmode, csm_norm = self._preprocess_csm(csm)
            strength_pred, loc_pred, noise_pred = self.model.predict(eigmode, verbose=0) # verbose = Terminalanzeige
            strength_pred = strength_pred.squeeze()
            strength_pred *= csm_norm
            
            loc_pred = self.pipeline.recover_loc(loc_pred.squeeze(), aperture=self.uma_config.mics.aperture)
            self.results = {
                'x': loc_pred[0].tolist(),
                'y': loc_pred[1].tolist(),
                's': strength_pred.tolist()
            }
            self._append_results()
            
    def start_model(self):
        print("Model process starting.")
        self.data_process = Process(target=self._yield_csm_to_queue, args=(self.model_stop_event,))
        self.data_process.start()
        print("Model process started.")
        try:
            self._predict()
        finally:
            self.stop_model()
            print("Model stopped.")
    
    def stop_model(self):
        print("Model process stopping.")
        self.model_stop_event.set()
        if self.data_process.is_alive():
            self.data_process.join()
        print("Model process stopped.")
        while not self.csm_queue.empty():
            self.csm_queue.get()
        print("Model queue empty.")   

    def _yield_beamforming_result_to_queue(self, stop_event):
        while not stop_event.is_set():
            # random data shape 64,64
            dummy = np.random.rand(64,64)
            self.beamforming_queue.put(dummy)
            time.sleep(0.1)
            
            #pm = self.bb.synthetic(self.frequency, 3)
            #Lm = ac.L_p(pm)
            #self.beamforming_queue.put(Lm)
            
    def _get_mean_beamforming(self):
        beamforming_list = []
        while len(beamforming_list) < self.min_csm_num and not self.beamforming_stop_event.is_set():
            if not self.beamforming_queue.empty():
                beamforming_list.append(self.beamforming_queue.get())
        
        if len(beamforming_list) > 0:
            csm_mean = np.mean(beamforming_list, axis=0)
            return csm_mean
        else:
            return None

    def start_beamforming(self):
        print("Beamforming process starting.")
        self.beamforming_process = Process(target=self._yield_beamforming_result_to_queue, args=(self.beamforming_stop_event,))
        self.beamforming_process.start()
        print("Beamforming process started.")
        try:
            self._get_mean_beamforming()
        finally:
            self.stop_beamforming()
            print("Beamforming stopped.")

    def stop_beamforming(self):
        print("Beamforming process stopping.")
        self.beamforming_stop_event.set()
        if self.beamforming_process.is_alive():
            self.beamforming_process.join()
            
        print("Beamforming process stopped.")
        while not self.beamforming_queue.empty():
            self.beamforming_queue.get()
        print("Beamforming queue empty.")
  
    def update_frequency(self, frequency):
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
    
