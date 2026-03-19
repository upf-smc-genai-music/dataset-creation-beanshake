# Beanshake Dataset 

This is a dataset containing 11 audio recordings (~40 seconds long each) of an alumninum water bottle being shook with a variable number of garbanzo beans inside. 
The key parameter in this dataset (BeanCount) is the number of garbanzo beans inside the water bottle.

**Created By:** Inés Broto Clemente and Ariana Pereira

## Extension: What acoustic features inform how an audio encodec learns rhythms in a controllable manner?

By witnessing that the audio wave form had a periodicity due to the steady shake rate, I decided to see if this could be learned as a parameter by the model so that when using this encodec during audio generation, a user can toggle the bean shake speed. 

To do this, I experimented with different feature extract methods and parameter manipulation, then trained an RNNeNcodec on the output. 

This repo contains the scripts to generate the .csv files used to train the [RNeNcodec](https://github.com/upf-smc-genai-music/classnn-pereiraa10).

_____
### Method 1: Amplitude Envelope

This method computes a smooth amplitude envelope from a waveform, resampled to the target sampling rate (75 frames per second). The steps include:

1. Mix to mono if stereo.
2. Full-wave rectify (abs value).
3. Low-pass filter to smooth rapid transients — cutoff tuned so that individual bean-hit peaks are captured but high-frequency noise is removed. `smoothing_ms` parameter is used here to control the Low-pass filter window size in milliseconds. Larger window size = smoother envelope; smaller window size = more transient detail. 
4. Downsample to target_fps by averaging within each frame window.

Code available in `amplitude.py`.

### Method 2: RMS in dB and Shake Rate in BPM

The aim of this method was to extract 2 features which would correspond with the shake intensity (rms_db) and shake speed (shake_rate_bpm)

**RMS in dB**: using the librosa library, I extracted frame-level RMS energy, resampled to target_fps and floored silence to -80 dB. 

**Shake Rate in BPM**: The approach includes:

1) Onset detection from librosa using a parameter `onset_delta`, higher delta = fewer, more confident onsets 
2) Convert to an interonset interval (IOI) time series 
3) Linearly interpolate IOI back onto the full frame grid with flat extrapolation at hoth edges (holds the nearest measured value)
4) Convert to BPM scale (BPM = 60 / IOI)

### Method 3: Improve Shake Rate in BPM extraction with Amplitude Envelope 

This method combines the first method into the calculation of `shake_rate_bpm`. BPM is deriving from peaks in the amplitude envelope, interpolated to a continuous signal. The steps include:

1) Build a smoothed amplitude envelope at the native sample rate using the `smoothing_ms` parameter as explained above.
2) Identify individual shake events by finding peaks in the envelope with a prominence threshold `peak_prominence` and a minimum inter-peak distance `min_ioi_ms` 
3) Compute IOI between peaks 
4) Linearly interpolate IOI
5) Convert to BPM scale 

### Rejected Method: 
In the `old_extract_features.py` script, you'll find methods to extract 6 audio features related to shake intensity and rhythm (amplitude_envelope, rms_db, spectral_flux, onset_strength, ioi_s, shake_rate_bpm). I decided not to use this for training the RNeNcodec since each of these features would result in a different toggle during generation inference, and some of these features are highly correlated with each other rendering them redundant. To make things simplier, I selected the 2 most relevant features (rms_db for intensity and shake_rate_bpm for tempo) to experiment with. 

_____

To see the results of these different approaches, check out the [RNeNcodec notebook](https://github.com/upf-smc-genai-music/)! 