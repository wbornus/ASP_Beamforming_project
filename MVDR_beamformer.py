import numpy as np
import matplotlib.pyplot as plt
import rir_generator as rir


def MVDR_beamformer(x, distances, phi, fs, mic_directivity="omni", rotations=None):
    """
    :param x: input time signals [M x N], M - number of microphones, N - number of samples
    :param distances: [0, d1, d2, ... dM] - distances between microphone 0 to mic i
    :param phi: direction of beam (scalar) [rad]
    :param fs: sampling frequency
    :param mic_directivity: directivity of microphones - omni (default) / cardioid
    :param rotations: rotation of each microphone in reference to front of matrix [rad]
    :return out: output signal
    """
    X = np.zeros([x.shape[0], len(np.fft.rfft(x[0, :]))]).astype(complex)
    for i in range(x.shape[0]):
        X[i, :] = np.fft.rfft(x[i, :])
    c = 343

    # steering vector
    d = np.zeros([X.shape[0], X.shape[1]]).astype(complex)
    # directivity-based weights
    if rotations is None:
        rotations = np.zeros(X.shape[0])
    U = np.ones(d.shape)
    if mic_directivity == "cardioid":
        for i in range(X.shape[0]):
            U[i, :] = 0.5*(1+np.cos(phi+rotations[i]))

    f = np.linspace(0, fs/2, X.shape[1])
    for i in range(X.shape[0]):
        tau = np.cos(phi) * distances[i] / c
        for j in range(X.shape[1]):
            d[i, j] = np.exp(-1j*2*np.pi*f[j]*tau)*U[i, j]

    R = np.zeros([X.shape[0], X.shape[0], X.shape[1]]).astype(complex)
    for i in range(X.shape[1]):
        for j in range(X.shape[0]):
            R[j, j, i] = 1+0*1j

    # calculation of covariance matrix
    # for i in range(X.shape[1]):
    #     for m in range(X.shape[0]):
    #         for n in range(X.shape[0]):
    #             # print(X[:, i])
    #             R[m, n, i] = np.var([X[m, i], np.conj(X[n, i])])
    #     print(R[:, :, i])

    hm = np.matrix(np.zeros([X. shape[0], X.shape[1]])).astype(complex)
    for i in range(X.shape[1]):
        dm = np.matrix(d[:, i])
        Rm = np.matrix(R[:, :, i])
        hm[:, i] = (np.linalg.inv(Rm)*dm.T)/(dm.H.T*np.linalg.inv(Rm)*dm.T)

    Ym = np.zeros(X.shape[1]).astype(complex)
    for i in range(X.shape[1]):
        Ym[i] = np.matrix(X[:, i])*hm[:, i]

    Y = np.array(Ym)
    out = np.fft.irfft(Y)
    return out


def polar_pattern(mics, angle):
    """
    :param mics: mics placements, should be around center of 10x10 room
    :param angle: angle of looking [degrees]
    :return out_power, theta: power vs angle
    """

    r = 4
    height = 1.2
    # rng = np.random.default_rng(1254)
    # sig = rng.random(10000)
    # test_signal = (sig - 0.5) * 2
    Fs = 22100
    distances = np.array([])
    print(np.size(mics, axis=0))
    for i in range(np.size(mics, axis=0)):
        distances = np.append(distances, np.sqrt((mics[0][0] - mics[i][0])**2 + (mics[0][1] - mics[i][1])**2))
    out_power = np.array([])
    phi_out = np.array([])
    for phi in np.linspace(0, np.pi*2, 200):
        phi_out = np.append(phi_out, phi)
        source = [r*np.cos(phi) + 5, r*np.sin(phi)+5, height]
        # Generating impulse respons using RIR and convolving with the signal
        n_samples = 300
        # signal = np.zeros(
        #     shape=(np.size(test_signal, axis=0) + n_samples - 1, 2, np.size(mics, axis=0)))

        h = rir.generate(
            c=343,  # Sound velocity (m/s)
            fs=Fs,  # Sample frequency (samples/s)
            r=mics,
            s=source,
            # Source position [x y z] (m)
            L=[10, 10, 3],  # Room dimensions [x y z] (m)
            reverberation_time=0.0,  # Reverberation time (s)
            nsample=n_samples,  # Number of output samples
            order=0,  # order of reflections
            mtype=rir.mtype.cardioid,
            orientation=[0, 0]
        )
        h_cor = h
        h_cor[:, 1] = h[:, 1] * (0.5 * (1 + np.cos(phi + np.pi / 2)))
        h_cor[:, 2] = h[:, 2] * (0.5 * (1 + np.cos(phi - np.pi / 2)))
        power = np.sqrt(np.mean(MVDR_beamformer(h_cor.T,
                                                distances,
                                                angle/360*2*np.pi,
                                                Fs,
                                                mic_directivity="cardioid",
                                                rotations=[0, np.pi/2, -np.pi/2])**2))
        out_power = np.append(out_power, power)
        print('angle: ', phi, power)

    return out_power, phi_out

# test
# do usunięcia później

# rng = np.random.default_rng(1254)
# sig = rng.random(1000)
# sig = (sig-0.5)*2
# # plt.plot(sig)
# # plt.show()
#
# Sig = np.zeros([3, 1000])
# for i in range(3):
#     Sig[i, :] = (rng.random(1000)-0.5)*2
#
# y = MVDR_beamformer(Sig, [0, 0.2, 0.4], 0, 1/8000)
# plt.plot(y)
# plt.show()

if __name__ == "__main__":
    room_d1 = 10
    room_d2 = 10
    room_d3 = 3
    mic_d_from_wall = 2.5

    #mics in front of the table
    # [x, y]
    mic_d = 0.3
    mic_h = 1.2
    mics = np.array([
        [room_d2/2, mic_d+room_d1/2, mic_h],
        [room_d2/2,  room_d1/2, mic_h],
        [room_d2/2,  -mic_d+room_d1/2, mic_h]
        ])

    [pattern, theta] = polar_pattern(mics, 90)
    # plt.plot(theta[250:750], 10*np.log10(pattern[250:750]/np.max(pattern)))
    # plt.show()

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.plot(theta, 10*np.log10(pattern/np.max(pattern)))
    plt.show()
