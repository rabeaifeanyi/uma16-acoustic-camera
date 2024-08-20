import numpy as np
import tensorflow as tf # type: ignore
import acoular as ac # type: ignore
from acoupipe.datasets.synthetic import DatasetSynthetic # type: ignore
from modelsdfg.transformer.config import ConfigBase # type: ignore
from.csm import CSM, calculate_combined_csm

center_frequencies = [250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000]

class ModelProcessor:
    """Model processor for the UMA-16 microphone array.
    """
    def __init__(self, frame_width, frame_height, uma_config, mic_index, model_dir, model_config_path, ckpt_path):
        """Initialize the model processor.
        
        Args:
            frame_width (int): Width of the frame.
            frame_height (int): Height of the frame.
            config (ConfigUMA): Configuration for the UMA-16 microphone array.
        """
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.mic_index = mic_index
        
        self.uma_config = uma_config
        
        self.dev = ac.SoundDeviceSamplesGenerator(device=self.mic_index, numchannels=16)
        self.recording_time = 1
        self.dev.numsamples = int(self.recording_time * self.dev.sample_freq)
        self.t = np.arange(self.dev.numsamples) / self.dev.sample_freq
        self.block_size = 128
        self.freqs = np.fft.rfftfreq(self.block_size, 1/self.dev.sample_freq)        
        
        self.model_dir = model_dir
        self.model_config_path = model_config_path
        self.ckpt_path = ckpt_path
        
        self._setup_model()
        
        # Dummy data, delete this later
        self.noise_level = 0.05
        self.dataset_uma = DatasetSynthetic(config=self.uma_config)
        self.data_generator = self.dataset_uma.generate(f=5000, 
                                                        num=3, 
                                                        size=100, 
                                                        split='validation', 
                                                        features = ['loc'], 
                                                        progress_bar=False)
    
    def get_uma_data(self):
        signal = ac.tools.return_result(self.dev, num=256)
        return {
            'x': self.t.tolist(), 
            'y': signal.tolist()
        }
        
    def _setup_model(self):
        model_config = ConfigBase.from_toml(self.model_config_path)
        self.pipeline = model_config.datasets[1].pipeline.create_instance()
        self.ref_mic_index = model_config.datasets[0].pipeline.args['ref_mic_index']
        model_config.datasets[1].validation.cache=False
        self.model = tf.keras.models.load_model(self.ckpt_path)
    
    def _prediction(self):
        csm, csm_norm = self._calc_csm()
        eigmode = self.model.preprocessing(csm[np.newaxis]).numpy()
        strength_pred, loc_pred, noise_pred = self.model.predict(eigmode)
        strength_pred = strength_pred.squeeze()
        strength_pred *= np.real(csm_norm)
        loc_pred = self.pipeline.recover_loc(loc_pred.squeeze(), aperture=self.uma_config.mics.aperture)
        return strength_pred, loc_pred, noise_pred
    
    def _calc_csm(self):
        # This is not the final CSM CALCULATION!! 
        # This just works with the model
        signal = ac.tools.return_result(self.dev, num=256)
        cnum_of_samples, num_of_channels = signal.shape
        fft_signal = np.fft.fft(signal, axis=0)
        csm = np.einsum('ij,ik->jk', fft_signal, np.conjugate(fft_signal)) / self.dev.numsamples
        csm_norm = csm[self.ref_mic_index, self.ref_mic_index]
        csm = csm / csm_norm
        return csm, csm_norm
        
    def _calc_csm_rabea(self):
        # TODO Fragen
        signal = ac.tools.return_result(self.dev, num=256)
        csm = CSM(signal, self.dev.sample_freq, block_size=self.block_size).calc_csm()
        csm = calculate_combined_csm(csm, self.freqs, center_frequencies)
        
        # at this point, the CSM is in [Volt^2] since the microphone channel data
        # is a voltage strea -> we need to convert it to squared sound pressure values [Pa^2]
        # we do not account for the true sensitivity of the MEMS microphones here! 
        # I just use a standard sensitivity of 16 mV/Pa for all microphones
        # TODO Fragen
        csm = csm / 0.0016**2
        
        # Normalize the CSM
        csm_norm = csm[self.ref_mic_index, self.ref_mic_index]
        csm = csm / csm_norm
        
        return csm, csm_norm
    
    # TODO mit Acoular umsetzen
    def _calc_csm_faster(self):
        
        # Adams Code einfügen!!!!!
        
        freq_data = ac.PowerSpectra(
            time_data=self.dev, 
            block_size=128, 
            window='Hanning'
            )
        f_ind = np.searchsorted(freq_data.fftfreq(), 800)
        csm = freq_data.csm[f_ind]             
        csm = csm / 0.0016**2
        csm_norm = csm[self.ref_mic_index, self.ref_mic_index]
        csm = csm / csm_norm
 
        return csm, csm_norm

    def uma16_ssl(self):
        strength_pred, loc_pred, noise_pred = self._prediction()
        
        print('\nx', loc_pred[0].tolist(), '\ny', loc_pred[1].tolist(), '\ns', strength_pred.tolist(), '\n')

        return {
            'x': loc_pred[0].tolist(),
            'y': loc_pred[1].tolist(),
            's': strength_pred.tolist()
        }

    def dummy_uma16_ssl(self):
        """Get dummy data for the UMA-16 microphone array.
        Delete this later
        """
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

        return {
            'x': noisy_loc[:, 0].tolist(),
            'y': noisy_loc[:, 1].tolist(),
            's': strengths.tolist()
        }
        