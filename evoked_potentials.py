from obci_readmanager.signal_processing.read_manager import ReadManager
from  scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.gridspec import GridSpec

def filtry(sygnal,Fs):
  #filtr notch 
  w0=50
  Q=25
  t=np.arange(0,len(sygnal)/Fs,1/Fs)
  b, a=signal.iirnotch(w0,Q,Fs)
  sygnal = signal.filtfilt(b,a,sygnal)

  #filtr dolnoprzepustowy butterwortcha
  N=2
  Wn=8
  b_b,a_b = signal.butter(N, Wn, btype = "lowpass", fs = Fs)
  sygnal = signal.filtfilt(b_b,a_b,sygnal,axis=0)

  #filtr górnoprzepustowy
  N2=2
  Wn2=1
  b_b2,a_b2 = signal.butter(N2, Wn2, btype = "highpass", fs = Fs)
  sygnal = signal.filtfilt(b_b2,a_b2,sygnal,axis=0)

  return t,sygnal

def usredniaj_tagi(sygnaly, tagi, sampling):
    lista=np.zeros((int(1.1*sampling)+1, sygnaly.shape[1], len(tagi)))
    
    for i, tag_start in enumerate(tagi):
        start=tag_start
        start_index = int(start*sampling-0.3*sampling)
        end_index = int(start*sampling-0.3*sampling)+int(1.1*sampling)
        lista[:,:,i]= sygnaly.loc[start_index:end_index]

        srednia = np.zeros((int(1.1*sampling)+1, sygnaly.shape[1]))
        for i in range(sygnaly.shape[1]):
            srednia[:,i] = np.sum(lista[:,i,:], axis = 1)
    srednia/=len(tagi)
    return srednia
      
def template_elektrody(sygnal, os, channels_names):
    fig = plt.figure(figsize=(10,8))
    gs = GridSpec(5, 5, width_ratios=[1, 1, 1, 1, 1], height_ratios=[1, 1, 1,1, 1])


    ax1 = plt.subplot(gs[0, 1])
    ax1.plot(os,sygnal[:,0])
    ax1.set_title(channels_names[0])

    ax2 = plt.subplot(gs[0, 3])
    ax2.plot(os,sygnal[:,1])
    ax2.set_title(channels_names[1])


    for i in range(3):
        for j in range(5):
            ax = plt.subplot(gs[i + 1, j])
            ax.plot(os,sygnal[:,i * 5 + j + 3 -1])
            ax.set_title(channels_names[i * 5 + j + 3 -1])
            #ax.set_title(f'Plot {i * 5 + j + 3}')


    ax3 = plt.subplot(gs[4, 1])
    ax3.plot(os,sygnal[:,17])
    ax3.set_title(channels_names[17])

    ax4 = plt.subplot(gs[4, 3])
    ax4.plot(os,sygnal[:,18])
    ax4.set_title(channels_names[18])

    # Adjust layout
    plt.tight_layout()
    plt.show()


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
    t,sygnal_przefiltrowany=filtry(sygnaly[:,i], sampling)
    sygnaly[:,i]=sygnal_przefiltrowany
'''
plt.plot(sygnaly[:,0])
plt.xlim(0,1000)
plt.show()
'''
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


srednia4=usredniaj_tagi(sygnaly, tag4, sampling)  

#print(lista1)  
t=np.arange(-0.3,0.8,1/sampling)
plt.plot(t,srednia4[:,16])
plt.show()

srednia3=usredniaj_tagi(sygnaly, tag3, sampling)  
plt.plot(t,srednia3[:,18])
plt.show()


#sredni potencjal we wszystkich kanalach dla 4 tagu
template_elektrody(srednia4, t, channels_names)

#wybranie 1 kanalu-Pz do identyfikacji zalamkow
wybrany_kanal=14
sygnal=srednia4[:,wybrany_kanal]
plt.plot(t,sygnal)
plt.show()

#amplituda P1 jako różnica między ekstremalną wartością załamka P1 a ekstremalną wartością załamka N2.
#wybranie P1
