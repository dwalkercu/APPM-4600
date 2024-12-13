'''
A library of general-use functions.
AUTHOR: Derek Walker
'''

import numpy as np

def awgn(signal, snr_db):
    """Returns the noisy signal using additive white Gaussian noise

    signal - the signal to add noise to
    snr_db - the signal-to-noise ratio in dB
    """
    # calculate signal power
    signal_power = np.mean(np.abs(signal)**2)

    # calculate noise power based on SNR
    noise_power = signal_power / (10**(snr_db/10))

    # generate Gaussian noise with zero mean with variance sqrt(noise_power)
    noise = np.random.normal(0, np.sqrt(noise_power), signal.shape)

    # make noisy signal
    return signal + noise
