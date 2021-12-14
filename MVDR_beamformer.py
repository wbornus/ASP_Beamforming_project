import numpy as np
import matplotlib.pyplot as plt


def MVDR_beamformer(x, distances, phi,  fs):
    """
    :param x: input time signals [M x N], M - number of microphones, N - number of samples
    :param distances: [0, d1, d2, ... dM] - distances between microphone 0 to mic i
    :param phi: direction of beam (scalar) [rad]
    :param fs: sampling frequency
    :return out: output signal
    """
    X = np.zeros([x.shape[0], len(np.fft.rfft(x[0, :]))]).astype(complex)
    for i in range(x.shape[0]):
        X[i, :] = np.fft.rfft(x[i, :])
    c = 343

    # steering vector
    d = np.zeros([X.shape[0], X.shape[1]]).astype(complex)
    f = np.linspace(0, fs/2, X.shape[1])
    for i in range(X.shape[0]):
        tau = np.cos(phi) * distances[i] / c
        for j in range(X.shape[1]):
            d[i, j] = np.exp(-1j*2*np.pi*f[j]*tau)

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


# test
# do usunięcia później

rng = np.random.default_rng(1254)
sig = rng.random(1000)
sig = (sig-0.5)*2
# plt.plot(sig)
# plt.show()

Sig = np.zeros([3, 1000])
for i in range(3):
    Sig[i, :] = (rng.random(1000)-0.5)*2

y = MVDR_beamformer(Sig, [0, 0.2, 0.4], 0, 1/8000)
plt.plot(y)
plt.show()
