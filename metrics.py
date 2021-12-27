import numpy as np
from MVDR_beamformer import MVDR_beamformer
import rir_generator as rir
from tqdm import tqdm
import matplotlib.pyplot as plt

simulation_params = {
    'room_dims': [10, 10, 3],
    'num_channels': 10,
    'room_reverb': 0.5,
    'mic_distance': 0.10,
    'beamformer_position': [3.5, 3.5, 1.5],
    'source_distance': 2.5,
    'angle_resolution': 64,
    'angle_of_looking': 0,
}

def SNR(audio):
    return NotImplemented


def cutoff_angle(frequency, cutoff_level, fs, time, simulation_params):
    """
    :param frequency:
    :param cutoff_level: what level to cross to return angle
    :return: angle at which cutoff level is crossed
    """
    # print('setting params')
    time_domain = np.linspace(0, time, int(fs*time))
    signal = np.cos(2*np.pi*frequency*time_domain) + 0.001*np.random.normal(loc=0, scale=0.25, size=time_domain.shape)
    # signal = np.random.normal(loc=0, scale=1, size=time_domain.shape)

    room_dims = simulation_params['room_dims']
    num_channels = simulation_params['num_channels']
    reverb = simulation_params['room_reverb']
    mic_distance = simulation_params['mic_distance']
    beamformer_position = simulation_params['beamformer_position']
    source_distance = simulation_params['source_distance']
    angle_resolution = simulation_params['angle_resolution']

    mics = []
    mic_offset = (num_channels*mic_distance)/2
    for it in range(num_channels):
        mics.append(
            [beamformer_position[0] - mic_offset + it*mic_distance,
             beamformer_position[1],
             beamformer_position[2]]
        )
    mics = np.array(mics)
    """simulation"""
    # print('simulation')
    angle_domain = np.linspace(-np.pi, np.pi, angle_resolution)
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
        # print('angle = ', angle)
        # print('source = ', source)
        # print('mics = ', mics)
        X = np.empty(shape=(num_channels, signal.shape[0]))
        for channel_it in range(num_channels):
            X[channel_it, :] = np.convolve(signal, h[:, channel_it], mode='same')
        distances = np.array([])
        for i in range(np.size(mics, axis=0)):
            distances = np.append(distances, np.sqrt((mics[0][0] - mics[i][0]) ** 2 + (mics[0][1] - mics[i][1]) ** 2))
        beamformer_output = MVDR_beamformer(x=X, distances=distances,
                                            phi=simulation_params['angle_of_looking']*2*np.pi/360,
                                            fs=fs)

        tmp_rms = np.sqrt(np.mean(beamformer_output**2))
        energies[angle_it] = tmp_rms
        # print(tmp_rms)
        # print('\n')
        # print('tmp rms = ', tmp_rms)

    # print('computing angle')
    max_energy = np.max(energies)
    angle_level = 20*np.log10(energies / max_energy)
    out = -1
    for it in range(len(angle_level)-1):
        if angle_level[it] > cutoff_level and angle_level[it+1] <=cutoff_level:
            out = angle_domain[it]
            break
    return out, angle_level


def angle_freq_response(fmin, fmax, f_resolution, simulation_params):
    frequencies = np.linspace(fmin, fmax, f_resolution)
    angles = np.linspace(-np.pi, np.pi, simulation_params['angle_resolution'])
    result_plot = np.zeros(shape=(frequencies.shape[0], angles.shape[0]))
    for it, frequency in enumerate(tqdm(frequencies)):
        _, plot = cutoff_angle(frequency=frequency,
                                   cutoff_level=-10,
                                   fs=16000,
                                   time=0.100,
                                   simulation_params=simulation_params)
        result_plot[it, :] = plot
    return frequencies, angles, result_plot

if __name__ == "__main__":
    # frequency = 1000
    # angle, plot = cutoff_angle(frequency=frequency,
    #                    cutoff_level=-10,
    #                    phi=np.pi/2,
    #                    fs=16000,
    #                    time=0.100,
    #                    simulation_params=simulation_params)
    #
    # plt.figure()
    # theta = np.linspace(0, 2*np.pi, simulation_params['angle_resolution'])
    # plt.polar(theta, plot)
    # plt.title('angle: %d\nfrequency: %d' %(simulation_params['angle_of_looking'], frequency))
    # plt.show()

    freqs, angles, plot = angle_freq_response(fmin=0, fmax=4000, f_resolution=64,
                                              simulation_params=simulation_params)

    plt.figure(figsize=(10, 10))
    plt.pcolormesh(angles, freqs, plot)
    plt.xlabel('angle (rad)')
    plt.ylabel('frequency (Hz)')
    plt.colorbar()
    plt.show()

