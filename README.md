# Beanshake Dataset 

This dataset contains 11 audio recordings (~40 seconds each) of an alumninum water bottle being shaken with a variable number of garbanzo beans inside. 
The key parameter in this dataset (BeanCount) is the number of garbanzo beans inside the water bottle.

**Created By:** Inés Broto Clemente and Ariana Pereira

## Final Project: What acoustic features can be used to train an audio encodec to learns shake tempo and shake intensity in a controllable mannner?

After observing that the audio waveform exhibits periodicity due to a steady shake rate, I explored whether this could be learned as a parameter by the model. The goal is to enable users to control the shake speed during audio generation using the EnCodec model.

To do this, I experimented with different feature extract methods and parameter manipulation, then trained an RNNeNcodec on the resulting features. 

This repository contains the scripts to generate the .csv files used to train the [RNeNcodec](https://github.com/upf-smc-genai-music/classnn-pereiraa10).

_____
### Method 1: Amplitude Envelope

This method computes a smooth amplitude envelope from a waveform, resampled to the target sampling rate (75 frames per second). The steps include:

1. Mix to mono if stereo.
2. Apply full-wave rectification (abs value).
3. Apply a low-pass filter to smooth rapid transients. The cutoff is tuned to capture individual bean-hit peaks while removing high-frequency noise. The `smoothing_ms` parameter controls the filter window size (in milliseconds): larger window size produces a smoother envelope, while smaller values preserve more transient detail. 
4. Downsample to target_fps by averaging within each frame window.

Code available in `amplitude.py`.

### Method 2: RMS in dB and Shake Rate in BPM

The aim of this method is to extract two features corresponding to shake intensity (`rms_db`) and shake speed (`shake_rate_bpm`)

**RMS in dB**: Using the librosa library, frame-level RMS energy is extracted, resampled to `target_fps` and silence is floored at -80 dB. 

**Shake Rate in BPM**: The approach includes:

1) Detect onsets using librosa and a parameter `onset_delta` (higher delta = fewer, more confident onsets). 
2) Convert to an inter-onset interval (IOI) time series 
3) Linearly interpolate of IOI back onto the full frame grid, with flat extrapolation at both edges (holds the nearest measured value)
4) Convert to BPM scale (BPM = 60 / IOI)

Code available in `rms_shake_bpm.py`

### Method 3: Improve Shake Rate in BPM extraction with Amplitude Envelope 

This method incorporates the amplitude envelope in Method 1 into the calculation of `shake_rate_bpm`. BPM is derived from peaks in the amplitude envelope and interpolated into a continuous signal. The steps include:

1) Construct a smoothed amplitude envelope at the native sample rate using the `smoothing_ms` parameter.
2) Identify individual shake events by detecting peaks in the envelope using a prominence threshold (`peak_prominence`) and a minimum inter-peak distance (`min_ioi_ms`)
3) Compute IOIs between peaks 
4) Linearly interpolate IOIs across time
5) Convert to BPM 

Code available in `improved_rms_shake_bpm.py`

### Rejected Method: 
The `rejected_method.py` script includes methods for extracting six audio features related to shake intensity and rhythm:
- amplitude_envelope
- rms_db
- spectral_flux
- onset_strength
- ioi_s
- shake_rate_bpm

I chose not to use this approach for training the RNeNcodec model because each feature would introduce a separate control parameter during generation, and several of these features are highly correlated, making them redundant.  

To simplify the system, I selected the two most relevant features, `rms_db`(intensity) and `shake_rate_bpm`(tempo) for experimentation. 

Code available in `rejected_method.py`
_____

To see the results of these approaches, check out the [RNeNcodec notebook](https://github.com/upf-smc-genai-music/)! 