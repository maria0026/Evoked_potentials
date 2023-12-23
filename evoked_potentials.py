from obci_readmanager.signal_processing.read_manager import ReadManager
from  scipy import signal
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
sygnaly = mgr.get_samples()
# Pobierz wszystkie znaczniki
tags = mgr.get_tags()

sygnaly=sygnaly.T
sygnaly*=0.07150000333786011
'''
plt.plot(sygnaly[:,0])
plt.xlim(0,1000)
plt.show()
'''
#filtrowanie
for i in range(0, sygnaly.shape[1]):
    sygnaly[:,i]=sygnaly[:,i]-((sygnaly[:,-2])+(sygnaly[:,-1]))/2
    t,sygnal_przefiltrowany=amplitude.filtry(sygnaly[:,i], sampling)
    sygnaly[:,i]=sygnal_przefiltrowany

sygnaly=pd.DataFrame(sygnaly)

names=['0.1', '0.33', '0.66', '1']

tagi, tag1, tag2, tag3, tag4 = np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
for tag in tags:
    tagi = np.append(tagi, tag["start_timestamp"])
    if tag["name"] == names[0]:
        tag1 = np.append(tag1, tag["start_timestamp"])
    elif tag["name"] == names[1]:
        tag2 = np.append(tag2, tag["start_timestamp"])
    elif tag["name"] == names[2]:
        tag3 = np.append(tag3, tag["start_timestamp"])
    elif tag["name"] == names[3]:
        tag4 = np.append(tag4, tag["start_timestamp"])

#print(len(tagi))
print(tag1[0:5])
print(tag4[0:5])
srednia1=amplitude.usredniaj_tagi(sygnaly, tag1, sampling)
srednia2=amplitude.usredniaj_tagi(sygnaly, tag2, sampling)
srednia3=amplitude.usredniaj_tagi(sygnaly, tag3, sampling)  
srednia4=amplitude.usredniaj_tagi(sygnaly, tag4, sampling)  

t=np.arange(-0.3,0.8,1/sampling)

#sredni potencjal we wszystkich kanalach dla 4 tagu
amplitude.template_elektrody(srednia4, t, channels_names)

#wybranie 1 kanalu-Pz do identyfikacji zalamkow
wybrany_kanal=14
sygnal1=srednia1[:,wybrany_kanal]
sygnal2=srednia2[:,wybrany_kanal]
sygnal3=srednia3[:,wybrany_kanal]
sygnal4=srednia4[:,wybrany_kanal]



p1_1, p1_index1, n2_1, n2_index1 = amplitude.find_p1_and_n2(sygnal1)
p1_2, p1_index2, n2_2, n2_index2 = amplitude.find_p1_and_n2(sygnal2)
p1_3, p1_index3, n2_3, n2_index3 = amplitude.find_p1_and_n2(sygnal3)
p1_4, p1_index4, n2_4, n2_index4 = amplitude.find_p1_and_n2(sygnal4)

#create a subplot
fig, ax = plt.subplots(2,2, figsize=(10,8))
ax[0,0].plot(t,sygnal1)
ax[0,0].scatter(t[p1_index1], p1_1, color='green')
ax[0,0].scatter(t[n2_index1], n2_1, color='red')
ax[0,0].set_title('Intensywność 0,1')
ax[0,0].set_ylabel('U [uV]')
ax[0,0].set_xlabel('t [s]')
ax[0,1].plot(t,sygnal2)
ax[0,1].scatter(t[p1_index2], p1_2, color='green')
ax[0,1].scatter(t[n2_index2], n2_2, color='red')
ax[0,1].set_title('Intensywność 0,33')
ax[0,1].set_xlabel('t [s]')
ax[0,1].set_ylabel('U [uV]')
ax[1,0].plot(t,sygnal3)
ax[1,0].scatter(t[p1_index3], p1_3, color='green')
ax[1,0].scatter(t[n2_index3], n2_3, color='red')
ax[1,0].set_title('Intensywność 0,66')
ax[1,0].set_xlabel('t [s]')
ax[1,0].set_ylabel('U [uV]')
ax[1,1].plot(t,sygnal4)
ax[1,1].scatter(t[p1_index4], p1_4, color='green')
ax[1,1].scatter(t[n2_index4], n2_4, color='red')
ax[1,1].set_title('Intensywność 1')
ax[1,1].set_xlabel('t [s]')
ax[1,1].set_ylabel('U [uV]')
plt.show()


#pierwszy sposób analizy - amplituda  P1 jako różnica między ekstremalną wartością załamka P1 a ekstremalną wartością załamka N2
amplituda1=p1_1-n2_1
amplituda2=p1_2-n2_2
amplituda3=p1_3-n2_3
amplituda4=p1_4-n2_4
print("Pierwszy sposób analizy")
print("Amplituda P1 1", amplituda1)
print("Amplituda P1 2", amplituda2)
print("Amplituda P1 3", amplituda3)
print("Amplituda P1 4", amplituda4)

#porównanie aplitud testem parametrycznym
#H0-amplitudy są takie same
#H1-amplitudy są różne

#Losowe przypisanie numerów warunków: hipoteza zerowa jest prawdziwa. 
#1 a 2
S=amplituda1-amplituda2
amplitude.test_permutacyjny(sygnaly, tag1, tag2, sampling, S, sposob_analizy=1)
#1 a 3
S=amplituda1-amplituda3
amplitude.test_permutacyjny(sygnaly, tag1, tag3, sampling, S, sposob_analizy=1)
#1 a 4
S=amplituda1-amplituda4
amplitude.test_permutacyjny(sygnaly, tag1, tag4, sampling, S, sposob_analizy=1)

#drugi sposób analizy - amplituda P1 jako amplituda P1 w stosunku do zera.
amplituda1=p1_1
amplituda2=p1_2
amplituda3=p1_3
amplituda4=p1_4
print("Drugi sposób analizy")
print("Amplituda P1 1", amplituda1)
print("Amplituda P1 2", amplituda2)
print("Amplituda P1 3", amplituda3)
print("Amplituda P1 4", amplituda4)


#1 a 2
S=amplituda1-amplituda2
amplitude.test_permutacyjny(sygnaly, tag1, tag2, sampling, S, sposob_analizy=2)
#1 a 3
S=amplituda1-amplituda3
amplitude.test_permutacyjny(sygnaly, tag1, tag3, sampling, S, sposob_analizy=2)
#1 a 4
S=amplituda1-amplituda4
amplitude.test_permutacyjny(sygnaly, tag1, tag4, sampling, S, sposob_analizy=2)


