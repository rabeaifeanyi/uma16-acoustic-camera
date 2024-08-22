import numpy as np
import tensorflow as tf # type: ignore
import acoular as ac # type: ignore
from multiprocessing import Process, Event, Queue
import time
from modelsdfg.transformer.config import ConfigBase # type: ignore
from acoupipe.datasets.synthetic import DatasetSynthetic # type: ignore
from .sd_generator import SoundDeviceSamplesGeneratorWithPrecision


####################################################################################################################################
# TODO Adam fragen: nicht so ganz verstanden wie BlockAverage funktioniert (ich benutze den csm_gen und mittel die CSMs selbst)
#                   -> Ist numaverage die Anzahl der zu mittelnden CSMs?
#                   -> Wie kann ich overflow sehen?
#                   -> Wenn numaverga=None, ist es dann das gleiche, w ie in meinem Code?
#         self.avg_csm = ac.BlockAverage(source=self.csm_gen, numaverage=10)

####################################################################################################################################


class ModelProcessor:
    def __init__(self, uma_config, mic_index, model_config_path, ckpt_path, buffer_size=20, min_csm_num=20):
        # general setup
        self.mic_index = mic_index
        self.uma_config = uma_config
        self.nfreqs = 257 
        
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
    
    # ___________________________________________________ MODEL FUNCTIONS ____________________________________________________   
    
    def _setup_model(self, model_config_path, ckpt_path):
        # load model
        model_config = ConfigBase.from_toml(model_config_path)
        self.pipeline = model_config.datasets[1].pipeline.create_instance()
        self.ref_mic_index = model_config.datasets[0].pipeline.args['ref_mic_index']
        model_config.datasets[1].validation.cache = False
        self.model = tf.keras.models.load_model(ckpt_path)
        
    def _yield_csm(self, stop_event):
        csm_counter = 0
        while not stop_event.is_set():
            data = next(self.csm_gen.result(num=self.nfreqs))
            data = data.reshape(self.csm_shape)
            data = data[60].reshape(self.dev.numchannels, self.dev.numchannels)
            self.csm_queue.put(np.real(data))
            csm_counter += 1
            
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
        
    def _preprocess_csm(self, csm):
        csm_norm = csm[self.ref_mic_index, self.ref_mic_index]
        eigmode = self.model.preprocessing(csm[np.newaxis]).numpy()
        return eigmode, csm_norm
    
    def _predict(self):
        num_predictions = 0
        while not self.stop_event.is_set():
            csm = self._get_mean_csm()
            if csm is None:
                continue
            num_predictions += 1
            eigmode, csm_norm = self._preprocess_csm(csm)
            strength_pred, loc_pred, noise_pred = self.model.predict(eigmode)
            strength_pred = strength_pred.squeeze()
            strength_pred *= np.real(csm_norm)
            
            loc_pred = self.pipeline.recover_loc(loc_pred.squeeze(), aperture=self.uma_config.mics.aperture)
            
            self.results = {
                'x': loc_pred[0].tolist(),
                'y': loc_pred[1].tolist(),
                's': strength_pred.tolist()
            }
    
    def start_model(self):
        self.data_process = Process(target=self._yield_csm, args=(self.stop_event,))
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
            print("Model stopped.")
    
    # __________________________________________________ EXTRA FUNCTIONS __________________________________________________   
         
    def get_uma_data(self):
        signal = ac.tools.return_result(self.dev, num=256)
        return {
            'x': self.t.tolist(), 
            'y': signal.tolist()
        }