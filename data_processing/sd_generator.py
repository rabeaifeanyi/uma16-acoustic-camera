import numpy as np
import sounddevice as sd # type: ignore
import acoular as ac # type: ignore

# converts to float64 before yielding
# Wenn das in CrossPowerSpectra gemacht wird, brauchen wir die Klasse nicht mehr

class SoundDeviceSamplesGeneratorWithPrecision(ac.SoundDeviceSamplesGenerator):
    def result(self, num):
        self.stream = stream_obj = sd.InputStream(
            device=self.device,
            channels=self.numchannels,
            clip_off=True,
            samplerate=self.sample_freq
        )
        with stream_obj as stream:
            self.running = True
            if self.numsamples == -1:
                while self.collectsamples:  # yield data as long as collectsamples is True
                    data, self.overflow = stream.read(num)
                    yield data[:num].astype(np.float64)

            elif self.numsamples > 0:  # amount of samples to collect is specified by user
                samples_count = 0  # numsamples counter
                while samples_count < self.numsamples:
                    anz = min(num, self.numsamples - samples_count)
                    data, self.overflow = stream.read(num)
                    yield data[:anz].astype(np.float64) 
                    samples_count += anz
        self.running = False
        return