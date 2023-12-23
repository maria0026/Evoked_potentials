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


def minima(sygnal):
    # Find all local minima
    #local_minima_indices = []
    local_minima_indices = np.where((sygnal[1:-1] < sygnal[:-2]) & (sygnal[1:-1] < sygnal[2:]))[0] + 1
    '''
    for i in range(1, len(sygnal) - 1):
        if sygnal[i] < sygnal[i - 1] and sygnal[i] < sygnal[i + 1]:
            local_minima_indices.append(i)
    '''
    # Extract values corresponding to local minima
    local_minima_values = sygnal[local_minima_indices]
    return local_minima_indices, local_minima_values

def maxima(sygnal):
    # Find all local minima
    local_maxima_indices = np.where((sygnal[1:-1] > sygnal[:-2]) & (sygnal[1:-1] > sygnal[2:]))[0] + 1
    #local_maxima_indices = []
    '''
    for i in range(1, len(sygnal) - 1):
        if sygnal[i] > sygnal[i - 1] and sygnal[i] > sygnal[i + 1]:
            local_maxima_indices.append(i)
    '''
    #Extract values corresponding to local minima
    local_maxima_values = sygnal[local_maxima_indices]
    return local_maxima_indices, local_maxima_values

def find_p1_and_n2(sygnal):
    local_minima_indices, local_minima_values = minima(sygnal)
    local_maxima_indices, local_maxima_values = maxima(sygnal)

    #Find P1
    p1 = np.max(local_maxima_values)
    # Return index of p1 in local maxima
    index = np.argmax(local_maxima_values)
    #Find index of p1
    p1_index = local_maxima_indices[index]
    #Find N2
    for indeks in local_minima_indices:
        if indeks > p1_index+15:
            n2_index = indeks
            break
    #Find n2
    n2 = sygnal[n2_index]
    return p1, p1_index, n2, n2_index


def test_permutacyjny(sygnaly, tag1, tag2, sampling, S_true, sposob_analizy=1, wybrany_kanal=14):
    #trzeba policzyć średnie potencjały z wymieszanych sygnałów oznaczonych tagiem np. 1 i 2
    #więc trzeba zmienić czym są tag- wylosować po trochę z 1 i 2
    roznica_amplitud = np.array([])
    for i in range(1000): #robimy 1000 iteracji 
        #take 50 elements from tag1 and 50 elements from tag2
        tag1_1 = np.random.choice(tag1, 50)
        tag1_1 = np.append(tag1_1, np.random.choice(tag2, 50))   
        potencjal_wywolany1=usredniaj_tagi(sygnaly, tag1_1, sampling)
        p1_1, p1_index1, n2_1, n2_index1 = find_p1_and_n2(potencjal_wywolany1[:,wybrany_kanal])
        if sposob_analizy==1:
            amplituda1=p1_1-n2_1
        else:
            amplituda1=p1_1

        tag2_2 = np.random.choice(tag1, 50)
        tag2_2 = np.append(tag2_2, np.random.choice(tag2, 50))
        potencjal_wywolany2=usredniaj_tagi(sygnaly, tag2_2, sampling)
        p1_2, p1_index2, n2_2, n2_index2 = find_p1_and_n2(potencjal_wywolany2[:,wybrany_kanal])
        if sposob_analizy==1:
            amplituda2=p1_2-n2_2
        else:
            amplituda2=p1_2

        S=amplituda2-amplituda1
        roznica_amplitud= np.append(roznica_amplitud, S)

    srednia_centyl = np.percentile(roznica_amplitud, 95)
    plt.hist(roznica_amplitud)
    plt.vlines(srednia_centyl, 0, 200, 'g')
    plt.vlines(S_true, 0, 200, 'r',label='Rzeczywista różnica amplitud')
    plt.xlabel('Różnica amplitud [uV]')
    plt.ylabel('Liczba wystąpień')
    plt.legend()
    plt.show()