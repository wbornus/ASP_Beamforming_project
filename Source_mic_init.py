#NumPy package
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from scipy.io import wavfile
from scipy.signal import firwin, lfilter

#init values
sample_rate_speech, speech_data = wavfile.read("recordings/mowa.wav")
sample_rate_noise, noise_data = wavfile.read("recordings/computer_fan.wav")
fs = sample_rate_speech
dt = 1/fs
c = 343



###Geometry setup
#mics in front of the table
# [x, y]
mic_d = 0.3
mics = np.array([
    [mic_d, 0],
    [0, 0],
    [-mic_d, 0]
    ])
mics_phi = 2*np.pi/360*np.array([0, 0, 0])

#Rectangle room 7 m x 4.6 m
room_d1 = 5
room_d2 = 3
mic_d_from_wall = 0.2
room_corners = np.array([
    [0-room_d2/2, -mic_d_from_wall],
    [0-room_d2/2, room_d1-mic_d_from_wall],
    [0+room_d2/2, room_d1-mic_d_from_wall],
    [0+room_d2/2, -mic_d_from_wall]
])

#Table in the middle of the room 3 m x 1.2 m
table_d_from_mic = 0.3
table_d1 = 3
table_d2 = 1.2
table_corners = np.array([
    [0-table_d2/2, table_d_from_mic],
    [0-table_d2/2, table_d1+table_d_from_mic],
    [0+table_d2/2, table_d1+table_d_from_mic],
    [0+table_d2/2, table_d_from_mic]
])

#sources 0-6 are the seven seats around the table, 7-x are noise sources (for example AC or fan noise from a hardware rack)
source_d_from_table = 0.1
rack_d_from_wall = 0.3
num_of_signal_sources = 6
num_of_noise_sources = 1
sources = np.array([
    [table_corners[0][0]-source_d_from_table, table_corners[0][1]+1*table_d1/6],
    [table_corners[0][0]-source_d_from_table, table_corners[0][1]+3*table_d1/6],
    [table_corners[0][0]-source_d_from_table, table_corners[0][1]+5*table_d1/6],
    [0, table_corners[1][1]+source_d_from_table],
    [table_corners[2][0]+source_d_from_table, table_corners[3][1]+5*table_d1/6],
    [table_corners[2][0]+source_d_from_table, table_corners[3][1]+3*table_d1/6],
    [table_corners[2][0]+source_d_from_table, table_corners[3][1]+1*table_d1/6],
    [room_corners[2][0]-rack_d_from_wall, room_corners[2][1]-rack_d_from_wall]
])



###geometry visualization
fig, ax1 = plt.subplots(figsize=(8, 8))
source_labels = range(0,np.size(sources, axis=0))
ax1.plot(mics[:, 0], mics[:, 1], 'ko', label='Microphones')
for i in range(0,np.size(mics, axis=0)):
    plt.annotate(i,
        (mics[i, 0],mics[i, 1]),
        textcoords="offset points", 
        xytext=(0,-12), 
        ha='center')

ax1.plot(sources[0:7, 0], sources[0:7, 1], 'rx', label='Sources (seats)')
for i in range(0,num_of_signal_sources+1):
    plt.annotate(i,
        (sources[i, 0],sources[i, 1]),
        textcoords="offset points", 
        xytext=(0,5), 
        ha='center')

ax1.plot(sources[7, 0], sources[7, 1], 'kx', label='Sources (noise)')
for i in range(num_of_signal_sources+1, num_of_signal_sources+num_of_noise_sources+1):
    plt.annotate(i-num_of_signal_sources-1,
        (sources[i, 0],sources[i, 1]),
        textcoords="offset points", 
        xytext=(0,5), 
        ha='center')

ax1.add_patch(Rectangle((table_corners[0][0], table_corners[0][1]), table_d2, table_d1,color="brown", edgecolor="black",label='Table'))
ax1.add_patch(Rectangle((room_corners[0][0], room_corners[0][1]), room_d2, room_d1,fill=False,edgecolor="black",label='Room'))
ax1.grid()
ax1.legend()
ax1.set_xlim([-room_d2/2-0.7, room_d2/2+1])
ax1.set_ylim([-mic_d_from_wall-0.7, room_d1+mic_d_from_wall+1])
ax1.set_title("Geometry of the conference setup")
plt.show()



###Propagation of the sound wave
def compute_delay(mic_coordinates, source_coordinates, c):
    """
    Parameters
    ----------
    mic_coordinates
        Coordinates of the microphone [x,y]
        
    source_coordinates
        Coordinates of the source [x,y]

    c
        Speed of sound [m/s]

    Returns
    -------
    t_delay
        Time delay for a specific microphone and source [s]

    p_delay
        Phase delay for a specific microphone and source [rad]

    """
    distance = np.sqrt((source_coordinates[0] - mic_coordinates[0])**2+(source_coordinates[1] - mic_coordinates[1])**2)
    t_delay = distance/c
    p_delay = 2*np.pi*30*t_delay
    return t_delay, p_delay


def compute_angle(mic_coordinates, source_coordinates):
    """
    Parameters
    ----------
    mic_coordinates
        Coordinates of the microphone [x,y]

    source_coordinates
        Coordinates of the source [x,y]

    Returns
    -------
    phi
        Angle from x axis to line between mic and source

    """
    phi = np.arctan2((source_coordinates[1] - mic_coordinates[1]), (source_coordinates[0] - mic_coordinates[0]))
    return phi

def directivity_gain(angle):
    """
    :param angle: angle from front of microphone (cardioid) [rad]
    :return: gain for sound from this direction [0-1]
    """
    return 0.5*(1 + np.cos(angle))



###Calculated delay, axis=0 is the specific microphone, axis=1 is the specific source, t_delay is the time delay, p_delay is the phase delay
t_delay = np.zeros(shape=(np.size(mics,axis=0), np.size(sources,axis=0)))
p_delay = np.zeros(shape=(np.size(mics,axis=0), np.size(sources,axis=0)))
for i in range(0,np.size(t_delay,axis=0)):
    for j in range(0,np.size(t_delay,axis=1)):
        t_delay[i,j], p_delay[i,j] = compute_delay(mics[i],sources[j],c)




###Speech data for mics. axis=0 is the mic number, axis=1 is the source number, axis=2 is the sample number.
s_delay = np.rint(t_delay*fs)
speech_data_for_mics = np.zeros((np.size(mics,axis=0),np.size(sources,axis=0),np.size(speech_data,axis=0)+int(s_delay.max())))
for i in range(0, np.size(speech_data_for_mics,axis=0)):
    for j in range(0, np.size(speech_data_for_mics,axis=1)):
        speech_data_for_mics[i,j,int(s_delay[i,j]):np.size(speech_data, axis=0)+int(s_delay[i,j])] += speech_data * directivity_gain(compute_angle(mics[i], sources[j]) + mics_phi[i])

##Plot example of a specific signal in a microphone
fig, ax2 = plt.subplots(figsize=(10, 8))
ax2.plot(speech_data_for_mics[0,3,:],'r', label="mic=0, source=3, t_delay = " + str(t_delay[0,3]))
ax2.plot(speech_data_for_mics[1,3,:],'b',label="mic=1, source=3, t_delay = " + str(t_delay[1,3]))
ax2.set_title("Speech signal for two different mics")
ax2.grid()
ax2.legend()
ax2.set_xlim([15600, 16100])
plt.show()
