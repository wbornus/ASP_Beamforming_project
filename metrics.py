import numpy as np
from MVDR_beamformer import MVDR_beamformer
import rir_generator as rir
from tqdm import tqdm
import matplotlib.pyplot as plt

simulation_params = {
    'room_dims': [10, 10, 3],
    'num_channels': 3,
    'room_reverb': 0.5,
    'mic_distance': 0.15,
    'beamformer_position': [3.5, 3.5, 1.5],
    'source_distance': 2,
    'angle_resolution': 100
}

def SNR(audio):
    return NotImplemented


def cutoff_angle(frequency, cutoff_level, phi, fs, time, simulation_params):
    """
    :param frequency:
    :param cutoff_level: what level to cross to return angle
    :return: angle at which cutoff level is crossed
    """
    print('setting params')
    time_domain = np.linspace(0, time, int(fs*time))
    # signal = np.cos(2*np.pi*frequency*time_domain) + 0.001*np.random.normal(loc=0, scale=0.25, size=time_domain.shape)
    signal = np.random.normal(loc=0, scale=1, size=time_domain.shape)

    room_dims = simulation_params['room_dims']
    num_channels = simulation_params['num_channels']
    reverb = simulation_params['room_reverb']
    mic_distance = simulation_params['mic_distance']
    beamformer_position = simulation_params['beamformer_position']
    source_distance = simulation_params['source_distance']
    angle_resolution = simulation_params['angle_resolution']

    mics = []
    mic_offset = num_channels*mic_distance
    for it in range(num_channels):
        mics.append(
            [beamformer_position[0],
             beamformer_position[1] - mic_offset + mic_distance,
             beamformer_position[2]]
        )
    mics = np.array(mics)
    """simulation"""
    print('simulation')
    angle_domain = np.linspace(0, 2*np.pi, angle_resolution)
    energies = np.zeros(shape=angle_domain.shape)
    for angle_it, angle in enumerate(angle_domain):
        source = [source_distance*np.cos(angle) + beamformer_position[0],
                  source_distance*np.sin(angle) + beamformer_position[1],
                  beamformer_position[2]]
        h = rir.generate(
            c=343,  # Sound velocity (m/s)
            fs=fs,  # Sample frequency (samples/s)
            r=mics,
            s=source,
            # Source position [x y z] (m)
            L=room_dims,  # Room dimensions [x y z] (m)
            reverberation_time=reverb,   # Reverberation time (s)
            nsample=1000,  # Number of output samples
            order=0,  # order of reflections
        )
        print('angle = ', angle)
        print('source = ', source)
        X = np.empty(shape=(num_channels, signal.shape[0]))
        for channel_it in range(num_channels):
            X[channel_it, :] = np.convolve(signal, h[:, channel_it], mode='same')
        distances = np.array([])
        for i in range(np.size(mics, axis=0)):
            distances = np.append(distances, np.sqrt((mics[0][0] - mics[i][0]) ** 2 + (mics[0][1] - mics[i][1]) ** 2))
        beamformer_output = MVDR_beamformer(x=X, distances=distances, phi=phi, fs=fs)

        tmp_rms = np.sqrt(np.mean(beamformer_output**2))
        energies[angle_it] = tmp_rms
        print(tmp_rms)
        print('\n')
        # print('tmp rms = ', tmp_rms)

    print('computing angle')
    max_energy = np.max(energies)
    angle_level = 20*np.log10(energies / max_energy)
    out = -1
    for it in range(len(angle_level)-1):
        if angle_level[it] > cutoff_level and angle_level[it+1] <=cutoff_level:
            out = angle_domain[it]
    return out, energies


def angle_frequency_map():
    return NotImplemented


if __name__ == "__main__":
    angle, plot = cutoff_angle(frequency=100,
                       cutoff_level=-10,
                       phi=90,
                       fs=16000,
                       time=0.100,
                       simulation_params=simulation_params)

    plt.figure()
    plt.plot(plot)
    plt.show()
