from obci_readmanager.signal_processing.read_manager import ReadManager
import  scipy.signal as sig
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.gridspec import GridSpec

def filtering(signal, Fs):
  #filtr notch 
  w0=50
  Q=25
  t=np.arange(0,len(signal)/Fs,1/Fs)
  b,a =sig.iirnotch(w0,Q,Fs)
  signal = sig.filtfilt(b,a,signal)

  #filtr dolnoprzepustowy butterwortcha
  N=2
  Wn=8
  b_b,a_b = sig.butter(N, Wn, btype = "lowpass", fs = Fs)
  signal = sig.filtfilt(b_b,a_b,signal,axis=0)

  #filtr górnoprzepustowy
  N2=2
  Wn2=1
  b_b2,a_b2 = sig.butter(N2, Wn2, btype = "highpass", fs = Fs)
  signal = sig.filtfilt(b_b2,a_b2,signal,axis=0)

  return t, signal

def average_by_tags(signals, tags, sampling):
    list_for_tagged_signals=np.zeros((int(1.1*sampling)+1, signals.shape[1], len(tags)))
    for i, tag_start in enumerate(tags):
        start=tag_start
        start_index = int(start*sampling-0.3*sampling)
        end_index = int(start*sampling-0.3*sampling)+int(1.1*sampling)
        list_for_tagged_signals[:,:,i]= signals.loc[start_index:end_index]

        evoked_potentials = np.zeros((int(1.1*sampling)+1, signals.shape[1]))
        for i in range(signals.shape[1]):
            evoked_potentials[:,i] = np.sum(list_for_tagged_signals[:,i,:], axis = 1)
    evoked_potentials/=len(tags)
    return evoked_potentials
      
def template_elektrodes(evoked_potentials, os, channels_names):
    fig = plt.figure(figsize=(10,8))
    gs = GridSpec(5, 5, width_ratios=[1, 1, 1, 1, 1], height_ratios=[1, 1, 1,1, 1])


    ax1 = plt.subplot(gs[0, 1])
    ax1.plot(os,evoked_potentials[:,0])
    ax1.set_title(channels_names[0])

    ax2 = plt.subplot(gs[0, 3])
    ax2.plot(os,evoked_potentials[:,1])
    ax2.set_title(channels_names[1])


    for i in range(3):
        for j in range(5):
            ax = plt.subplot(gs[i + 1, j])
            ax.plot(os,evoked_potentials[:,i * 5 + j + 3 -1])
            ax.set_title(channels_names[i * 5 + j + 3 -1])
            #ax.set_title(f'Plot {i * 5 + j + 3}')


    ax3 = plt.subplot(gs[4, 1])
    ax3.plot(os,evoked_potentials[:,17])
    ax3.set_title(channels_names[17])

    ax4 = plt.subplot(gs[4, 3])
    ax4.plot(os,evoked_potentials[:,18])
    ax4.set_title(channels_names[18])

    # Adjust layout
    plt.tight_layout()
    plt.show()


def minima(evoked_potential):
    # Find all local minima
    local_minima_indices = np.where((evoked_potential[1:-1] < evoked_potential[:-2]) & (evoked_potential[1:-1] < evoked_potential[2:]))[0] + 1
    # Extract values corresponding to local minima
    local_minima_values = evoked_potential[local_minima_indices]
    return local_minima_indices, local_minima_values

def maxima(evoked_potential):
    # Find all local maxima
    local_maxima_indices = np.where((evoked_potential[1:-1] > evoked_potential[:-2]) & (evoked_potential[1:-1] > evoked_potential[2:]))[0] + 1
    #Extract values corresponding to local minima
    local_maxima_values = evoked_potential[local_maxima_indices]
    return local_maxima_indices, local_maxima_values

def find_p1_and_n2(evoked_potential):
    local_minima_indices, local_minima_values = minima(evoked_potential)
    local_maxima_indices, local_maxima_values = maxima(evoked_potential)
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
    n2 = evoked_potential[n2_index]
    return p1, p1_index, n2, n2_index


def permutation_test(signals, tag1, tag2, sampling, S_true, method_of_analysis=1, chosen_channel=14):
    #trzeba policzyć średnie potencjały z wymieszanych sygnałów oznaczonych tagiem np. 1 i 2
    #więc trzeba zmienić czym są tag- wylosować po trochę z 1 i 2
    amplitude_differences = np.array([])
    for i in range(1000): #robimy 1000 iteracji 
        #take 50 elements from tag1 and 50 elements from tag2
        tag1_1 = np.random.choice(tag1, 50)
        tag1_1 = np.append(tag1_1, np.random.choice(tag2, 50))   
        evoked_potential1=average_by_tags(signals, tag1_1, sampling)
        p1_1, p1_index1, n2_1, n2_index1 = find_p1_and_n2(evoked_potential1[:,chosen_channel])
        if method_of_analysis==1:
            amplitude1=p1_1-n2_1
        else:
            amplitude1=p1_1

        tag2_2 = np.random.choice(tag1, 50)
        tag2_2 = np.append(tag2_2, np.random.choice(tag2, 50))
        evoked_potential2=average_by_tags(signals, tag2_2, sampling)
        p1_2, p1_index2, n2_2, n2_index2 = find_p1_and_n2(evoked_potential2[:,chosen_channel])
        if method_of_analysis==1:
            amplitude2=p1_2-n2_2
        else:
            amplitude2=p1_2

        S=amplitude2-amplitude1
        amplitude_differences= np.append(amplitude_differences, S)

    percentile = np.percentile(amplitude_differences, 95)
    plt.hist(amplitude_differences)
    plt.vlines(percentile, 0, 200, 'g')
    plt.vlines(S_true, 0, 200, 'r',label='Rzeczywista różnica amplitud')
    plt.xlabel('Różnica amplitud [uV]')
    plt.ylabel('Liczba wystąpień')
    plt.legend()
    plt.show()