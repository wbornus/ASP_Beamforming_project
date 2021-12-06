import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt


def crosscorrelation(channel1, channel2):
    corr1 = np.correlate(channel1, channel2, mode='full')
    corr2 = np.correlate(channel2, channel1, mode='full')

    
    return


filepath = 'person6.wav'

fs, audio = wavfile.read(filepath)
audio = audio/(2**15)

# print(fs)
# plt.plot(audio)
# plt.show()

c = 343
offset = 200
arbitrary_time_delay = (np.sqrt(41) - np.sqrt(17))/c
arbitrary_sample_delay = int(arbitrary_time_delay*fs)

print(f'arbitrary time delay = {arbitrary_time_delay}\narbitrary sample delay = {arbitrary_sample_delay}')

mic1 = audio
mic2 = np.concatenate((np.zeros(arbitrary_sample_delay, ), audio), axis=0)

mic1 = mic1[offset:offset+int(0.0125*fs)]
mic2 = mic2[offset:offset+int(0.0125*fs)]

plt.subplot(2, 1, 1)
plt.plot(mic1)
plt.subplot(2, 1, 2)
plt.plot(mic2)
plt.show()



corr, _ = crosscorrelation(mic1, mic1)
corr = corr[int(len(corr)/2)-1:]
# for frame in
plt.subplot(2, 1, 1)
plt.plot(mic1)
plt.subplot(2, 1, 2)
plt.plot(corr)
plt.title('autocorr')
plt.show()
print(f'mic 1 size = {len(mic1)}')

corr, _ = crosscorrelation(mic1, mic2)
# corr = corr[int(len(corr)/2)-1:]
# for frame in
plt.subplot(3, 1, 1)
plt.plot(mic1)
plt.subplot(3, 1, 2)
plt.plot(mic2)
plt.subplot(3, 1, 3)
plt.plot(corr)
plt.title('crosscorr')
plt.show()

noise = np.random.normal(loc=0, scale=0.25, size=(int(fs*0.0125), ))
_, corr = crosscorrelation(noise, noise)
plt.subplot(2, 1, 1)
plt.plot(noise)
plt.subplot(2, 1, 2)
plt.plot(corr[198:])
plt.title('noise autocorr')
plt.show()

