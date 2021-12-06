import numpy as np
import matplotlib.pyplot as plt

def MVDR_beamformer(x, distances, phi,  fs):
    """
    :param x: input time signals [M x N], M - number of microphones, N - number of samples
    :param distances: [0, d1, d2, ... dM] - distances between microphone 0 to mic i
    :param phi: direction of beam (scalar) [rad]
    :param fs: sampling frequency
    :return: output signal
    """
    X = np.fft.rfft(x)
    c = 343
    print(max(distances)/(c*fs))
    print(X.shape)

    # steering vector
    d = np.zeros([X.shape[0], X.shape[1]]).astype(complex)
    f = np.linspace(0, fs/2, X.shape[1])
    for i in range(X.shape[0]):
        tau = np.cos(phi) * distances[i] / c
        for j in range(X.shape[1]):
            d[i, j] = np.exp(-1j*2*np.pi*f[j]*tau)

    print(np.array(np.matrix(X).H).shape)
    print(X.shape)
    # covariance matrix
    R = np.zeros([X.shape[0], X.shape[0], X.shape[1], X.shape[1]]).astype(complex)
    for i in range(X.shape[0]):
        for j in range(X.shape[0]):
            Rij = np.cov(X[i, :], np.squeeze(np.array(np.matrix(X[j, :]).H)))
            # print(X[i, :])
            # print(np.squeeze(np.array(np.matrix(X[j, :]).H).T))
            R[i, j, :, :] = Rij

    print(R)

    return NotImplemented


# test

rng = np.random.default_rng(1254)
sig = rng.random(1000)
sig = (sig-0.5)*2
# plt.plot(sig)
# plt.show()

Sig = np.zeros([3, 1000])
for i in range(3):
    Sig[i, :] = sig

MVDR_beamformer(Sig, [0, 0.1, 0.2], 0.1, 1/8000)
