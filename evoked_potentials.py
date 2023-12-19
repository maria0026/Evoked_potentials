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

srednia1=usredniaj_tagi(sygnaly, tag1, sampling)
srednia2=usredniaj_tagi(sygnaly, tag2, sampling)
srednia3=usredniaj_tagi(sygnaly, tag3, sampling)  
srednia4=usredniaj_tagi(sygnaly, tag4, sampling)  

t=np.arange(-0.3,0.8,1/sampling)

#sredni potencjal we wszystkich kanalach dla 4 tagu
template_elektrody(srednia4, t, channels_names)

#wybranie 1 kanalu-Pz do identyfikacji zalamkow
wybrany_kanal=14
sygnal1=srednia1[:,wybrany_kanal]
sygnal2=srednia2[:,wybrany_kanal]
sygnal3=srednia3[:,wybrany_kanal]
sygnal4=srednia4[:,wybrany_kanal]

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


p1_1, p1_index1, n2_1, n2_index1 = find_p1_and_n2(sygnal1)
p1_2, p1_index2, n2_2, n2_index2 = find_p1_and_n2(sygnal2)
p1_3, p1_index3, n2_3, n2_index3 = find_p1_and_n2(sygnal3)
p1_4, p1_index4, n2_4, n2_index4 = find_p1_and_n2(sygnal4)

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

#porównanie aplitud testem parametrycznym
#H0-amplitudy są takie same
#H1-amplitudy są różne

#Losowe przypisanie numerów warunków: hipoteza zerowa jest prawdziwa. 
def test_permutacyjny(sygnaly, tag1, tag2, sampling, S_true):
    wybrany_kanal=14
    #trzeba policzyć średnie potencjały z wymieszanych sygnałów oznaczonych tagiem np. 1 i 2
    #więc trzeba zmienić czym są tag- wylosować po trochę z 1 i 2
    roznica_amplitud = np.array([])
    for i in range(1000): #robimy 1000 iteracji 
        #take 50 elements from tag1 and 50 elements from tag2
        tag1_1 = np.random.choice(tag1, 50)
        tag1_1 = np.append(tag1_1, np.random.choice(tag2, 50))   
        potencjal_wywolany1=usredniaj_tagi(sygnaly, tag1_1, sampling)
        p1_1, p1_index1, n2_1, n2_index1 = find_p1_and_n2(potencjal_wywolany1[:,wybrany_kanal])
        
        amplituda1=p1_1-n2_1
        #print(amplituda1)
        tag2_2 = np.random.choice(tag1, 50)
        tag2_2 = np.append(tag2_2, np.random.choice(tag2, 50))
        potencjal_wywolany2=usredniaj_tagi(sygnaly, tag2_2, sampling)
        p1_2, p1_index2, n2_2, n2_index2 = find_p1_and_n2(potencjal_wywolany2[:,wybrany_kanal])
        amplituda2=p1_2-n2_2
        #print(amplituda2)
        S=amplituda2-amplituda1
        #print(S)
        roznica_amplitud= np.append(roznica_amplitud, S)

    srednia_centyl = np.percentile(roznica_amplitud, 95)
    plt.hist(roznica_amplitud)
    plt.vlines(srednia_centyl, 0, 200, 'g')
    plt.vlines(S_true, 0, 200, 'r',label='Rzeczywista różnica amplitud')
    plt.xlabel('Różnica amplitud [uV]')
    plt.ylabel('Liczba wystąpień')
    plt.legend()
    plt.show()
#inny sposob
amplituda1=p1_1-n2_1
amplituda2=p1_2-n2_2
amplituda3=p1_3-n2_3
amplituda4=p1_4-n2_4

#test_permutacyjny(sygnaly, tag1, tag4, sampling, S)
#1 a 2
S=amplituda1-amplituda2
test_permutacyjny(sygnaly, tag1, tag2, sampling, S)
print("1,2")
#1 a 3
S=amplituda1-amplituda3
test_permutacyjny(sygnaly, tag1, tag3, sampling, S)
print("1,3")
#1 a 4
S=amplituda1-amplituda4
test_permutacyjny(sygnaly, tag1, tag4, sampling, S)

