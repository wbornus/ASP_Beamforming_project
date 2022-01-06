import numpy as np
import scipy.signal as ss
import rir_generator as rir
from scipy.io import wavfile
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import json

#init values
#Mowa.wav is 24-bit and is normalized to 0dBFS
sample_rate_speech, speech_data = wavfile.read("recordings/mowa_6,5s.wav")
#computer_fan.wav is 24-bit and is normalized to -2dBFS
sample_rate_noise, noise_data = wavfile.read("recordings/computer_fan_6,5s.wav")

###Geometry setup
#Rectangle room 7 m x 4.6 m
room_d1 = 5
room_d2 = 3
room_d3 = 3
room_corners = np.array([
    [0, 0],
    [0, room_d1],
    [0+room_d2, room_d1],
    [0+room_d2, 0]
])

#Table in the middle of the room 3 m x 1.2 m
table_d_from_mic = 0.3
table_d1 = 3
table_d2 = 1.2
table_corners = np.array([
    [0-table_d2/2+room_d2/2, table_d_from_mic],
    [0-table_d2/2+room_d2/2, table_d1+table_d_from_mic],
    [0+table_d2/2+room_d2/2, table_d1+table_d_from_mic],
    [0+table_d2/2+room_d2/2, table_d_from_mic]
])


#sources 0-6 are the seven seats around the table, 7-x are noise sources (for example AC or fan noise from a hardware rack)
source_d_from_table = 0.1
source_h = 1.2
rack_d_from_wall = 0.3
sources = np.array([
    [table_corners[0][0]-source_d_from_table, table_corners[0][1]+1*table_d1/6, source_h],
    [table_corners[0][0]-source_d_from_table, table_corners[0][1]+3*table_d1/6, source_h],
    [table_corners[0][0]-source_d_from_table, table_corners[0][1]+5*table_d1/6, source_h],
    [(table_corners[0][0]+table_corners[2][0])/2, table_corners[1][1]+source_d_from_table, source_h],
    [table_corners[2][0]+source_d_from_table, table_corners[3][1]+5*table_d1/6, source_h],
    [table_corners[2][0]+source_d_from_table, table_corners[3][1]+3*table_d1/6, source_h],
    [table_corners[2][0]+source_d_from_table, table_corners[3][1]+1*table_d1/6, source_h],
])
noise_sources = np.array(
    [[room_corners[2][0]-rack_d_from_wall, room_corners[2][1]-rack_d_from_wall, 2*source_h]
    ])


#mics in front of the table
# [x, y]
mic_d = 0.05
mic_d_from_wall = table_corners[0][1]+3*table_d1/6
mic_h = 1
mics = np.array([
    [mic_d+room_d2/2, mic_d_from_wall, mic_h],
    [room_d2/2,  mic_d_from_wall, mic_h],
    [-mic_d+room_d2/2,  mic_d_from_wall, mic_h]
    ])
mics_phi = 2*np.pi/360*np.array([0, 0, 0])

mic_orientation = [ #orientation of mics in radians
    [np.pi/2, 0],
    [np.pi/2, 0],
    [np.pi/2, 0]
]

##geometry visualization
fig, ax1 = plt.subplots(figsize=(8, 8))
source_labels = range(0,np.size(sources, axis=0))
ax1.plot(mics[:, 0], mics[:, 1], 'ko', label='Microphones')
for i in range(0,np.size(mics, axis=0)):
    plt.annotate(i,
        (mics[i, 0],mics[i, 1]),
        textcoords="offset points", 
        xytext=(0,-12), 
        ha='center')

ax1.plot(sources[0:-1, 0], sources[0:-1, 1], 'rx', label='Sources (seats)')
for i in range(0,np.size(sources, axis=0)):
    plt.annotate(i,
        (sources[i, 0],sources[i, 1]),
        textcoords="offset points", 
        xytext=(0,5), 
        ha='center')

ax1.plot(noise_sources[0,0], noise_sources[0, 1], 'kx', label='Sources (noise)')
for i in range(0, np.size(noise_sources, axis=0)):
    plt.annotate(i,
        (noise_sources[i, 0],noise_sources[i, 1]),
        textcoords="offset points", 
        xytext=(0,5), 
        ha='center')

ax1.add_patch(Rectangle((table_corners[0][0], table_corners[0][1]), table_d2, table_d1,color="brown", edgecolor="black",label='Table'))
ax1.add_patch(Rectangle((room_corners[0][0], room_corners[0][1]), room_d2, room_d1,fill=False,edgecolor="black",label='Room'))
ax1.grid()
ax1.legend()
ax1.set_xlim([-0.7, room_d2+1])
ax1.set_ylim([-0.7, room_d1+1])
ax1.set_title("Geometry of the conference setup")
plt.show()



#Generating impulse respons using RIR and convolving with the signal
n_samples = 30000


signal = np.zeros(shape=(np.size(speech_data,axis=0)+n_samples-1, np.size(mics, axis=0), np.size(sources, axis=0)))

for i in range(0,np.size(mics, axis=0)):
    h_noise = rir.generate(
    c=343,                  # Sound velocity (m/s)
    fs=sample_rate_speech,                  # Sample frequency (samples/s)
    r=mics[i],
    s=noise_sources[0],    
        # Source position [x y z] (m)
    L=[room_d2, room_d1, room_d3],            # Room dimensions [x y z] (m)
    reverberation_time=0.8, # Reverberation time (s)
    nsample=n_samples, # Number of output samples
    order=5,           # order of reflections
    mtype=rir.mtype.cardioid,
    orientation=mic_orientation[i]
    )

    for j in range(0, np.size(sources, axis=0)):
        h = rir.generate(
            c=343,                  # Sound velocity (m/s)
            fs=sample_rate_speech,                  # Sample frequency (samples/s)
            r=mics[i],
            s=sources[j],    
                # Source position [x y z] (m)
            L=[room_d2, room_d1, room_d3],            # Room dimensions [x y z] (m)
            reverberation_time=0.8, # Reverberation time (s)
            nsample=n_samples, # Number of output samples
            order=5,           # order of reflections
            mtype=rir.mtype.cardioid,
            orientation=mic_orientation[i]
        )


        # Convolve signal with impulse responses and add noise from the noise source
        signal[:,i,j] = ss.convolve(h[:, 0], speech_data[:, 0]) + ss.convolve(h_noise[:,0], noise_data[:,0])
        
        

#[samples, stereo channel, reciever index, source index]
print(signal.shape) 

fig, ax2 = plt.subplots(figsize=(10, 8))
ax2.plot(signal[:,1,0], 'r', label="reciever 1, source 0")
ax2.plot(signal[:,1,3], 'b', label="reciever 1, source 3")
ax2.plot(signal[:,1,5], 'g', label="reciever 1, source 5")
ax2.legend()
plt.show()

np.save('signal', signal)