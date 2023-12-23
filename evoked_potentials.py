from obci_readmanager.signal_processing.read_manager import ReadManager
import  scipy.signal as sig
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.gridspec import GridSpec
import amplitude


mgr = ReadManager("Karol_VEP.obci.xml", "Karol_VEP.obci.raw", "Karol_VEP.obci.tag")

# Pobierz informacje o sygnale
sampling = float(mgr.get_param("sampling_frequency"))
num_of_channels = int(mgr.get_param("number_of_channels"))
channels_names = mgr.get_param("channels_names")

# Pobierz cały sygnał
signals = mgr.get_samples()
# Pobierz wszystkie znaczniki
tags = mgr.get_tags()

signals=signals.T
signals*=0.07150000333786011
'''
plt.plot(signals[:,0])
plt.xlim(0,1000)
plt.show()
'''
#filtering
for i in range(0, signals.shape[1]):
    signals[:,i]=signals[:,i]-((signals[:,-2])+(signals[:,-1]))/2
    t,signal_filtered=amplitude.filtering(signals[:,i], sampling)
    signals[:,i]=signal_filtered

signals=pd.DataFrame(signals)
names=['0.1', '0.33', '0.66', '1']

all_tags, tag1, tag2, tag3, tag4 = np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
for tag in tags:
    all_tags = np.append(all_tags, tag["start_timestamp"])
    if tag["name"] == names[0]:
        tag1 = np.append(tag1, tag["start_timestamp"])
    elif tag["name"] == names[1]:
        tag2 = np.append(tag2, tag["start_timestamp"])
    elif tag["name"] == names[2]:
        tag3 = np.append(tag3, tag["start_timestamp"])
    elif tag["name"] == names[3]:
        tag4 = np.append(tag4, tag["start_timestamp"])

evoked_potentials_1=amplitude.average_by_tags(signals, tag1, sampling)
evoked_potentials_2=amplitude.average_by_tags(signals, tag2, sampling)
evoked_potentials_3=amplitude.average_by_tags(signals, tag3, sampling)  
evoked_potentials_4=amplitude.average_by_tags(signals, tag4, sampling)  

#time axis
t=np.arange(-0.3,0.8,1/sampling)

#the mean potential in all channels for 4 tag
amplitude.template_elektrodes(evoked_potentials_4, t, channels_names)

#choose 1 channel-Pz to identify peaks
chosen_channel=14
evoked_potential_1=evoked_potentials_1[:,chosen_channel]
evoked_potential_2=evoked_potentials_2[:,chosen_channel]
evoked_potential_3=evoked_potentials_3[:,chosen_channel]
evoked_potential_4=evoked_potentials_4[:,chosen_channel]

p1_1, p1_index1, n2_1, n2_index1 = amplitude.find_p1_and_n2(evoked_potential_1)
p1_2, p1_index2, n2_2, n2_index2 = amplitude.find_p1_and_n2(evoked_potential_2)
p1_3, p1_index3, n2_3, n2_index3 = amplitude.find_p1_and_n2(evoked_potential_3)
p1_4, p1_index4, n2_4, n2_index4 = amplitude.find_p1_and_n2(evoked_potential_4)

#create a subplot
fig, ax = plt.subplots(2,2, figsize=(10,8))
ax[0,0].plot(t,evoked_potential_1)
ax[0,0].scatter(t[p1_index1], p1_1, color='green')
ax[0,0].scatter(t[n2_index1], n2_1, color='red')
ax[0,0].set_title('Intensywność 0,1')
ax[0,0].set_ylabel('U [uV]')
ax[0,0].set_xlabel('t [s]')
ax[0,1].plot(t,evoked_potential_2)
ax[0,1].scatter(t[p1_index2], p1_2, color='green')
ax[0,1].scatter(t[n2_index2], n2_2, color='red')
ax[0,1].set_title('Intensywność 0,33')
ax[0,1].set_xlabel('t [s]')
ax[0,1].set_ylabel('U [uV]')
ax[1,0].plot(t,evoked_potential_3)
ax[1,0].scatter(t[p1_index3], p1_3, color='green')
ax[1,0].scatter(t[n2_index3], n2_3, color='red')
ax[1,0].set_title('Intensywność 0,66')
ax[1,0].set_xlabel('t [s]')
ax[1,0].set_ylabel('U [uV]')
ax[1,1].plot(t,evoked_potential_4)
ax[1,1].scatter(t[p1_index4], p1_4, color='green')
ax[1,1].scatter(t[n2_index4], n2_4, color='red')
ax[1,1].set_title('Intensywność 1')
ax[1,1].set_xlabel('t [s]')
ax[1,1].set_ylabel('U [uV]')
plt.show()

#first way of analysis - amplitude of P1 as a difference between the extreme value of P1 and the extreme value of N2
amplitude1=p1_1-n2_1
amplitude2=p1_2-n2_2
amplitude3=p1_3-n2_3
amplitude4=p1_4-n2_4
print("Pierwszy sposób analizy")
print("Amplituda P1 1", amplitude1)
print("Amplituda P1 2", amplitude2)
print("Amplituda P1 3", amplitude3)
print("Amplituda P1 4", amplitude4)

#comparing amplitudes with parametric test
#H0-amplitudes are the same
#H1-amplitudes are different
#random assignment of condition numbers: the null hypothesis is true.
#1 a 2
S=amplitude1-amplitude2
amplitude.permutation_test(signals, tag1, tag2, sampling, S, method_of_analysis=1)
#1 a 3
S=amplitude1-amplitude3
amplitude.permutation_test(signals, tag1, tag3, sampling, S, method_of_analysis=1)
#1 a 4
S=amplitude1-amplitude4
amplitude.permutation_test(signals, tag1, tag4, sampling, S, method_of_analysis=1)

#drugi sposób analizy - amplituda P1 jako amplituda P1 w stosunku do zera.
amplitude1=p1_1
amplitude2=p1_2
amplitude3=p1_3
amplitude4=p1_4
print("Drugi sposób analizy")
print("Amplituda P1 1", amplitude1)
print("Amplituda P1 2", amplitude2)
print("Amplituda P1 3", amplitude3)
print("Amplituda P1 4", amplitude4)


#1 a 2
S=amplitude1-amplitude2
amplitude.permutation_test(signals, tag1, tag2, sampling, S, method_of_analysis=2)
#1 a 3
S=amplitude1-amplitude3
amplitude.permutation_test(signals, tag1, tag3, sampling, S, method_of_analysis=2)
#1 a 4
S=amplitude1-amplitude4
amplitude.permutation_test(signals, tag1, tag4, sampling, S, method_of_analysis=2)


