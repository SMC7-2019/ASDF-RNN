import os

import IPython.display as ipd
from IPython.display import clear_output

import numpy as np
import matplotlib.pyplot as plt
import scipy

import librosa as librosa


def get_file_list(base_folder, extension='wav'):
    
    file_list = os.listdir(base_folder)
    file_list_filtered = []
    
    for filename in file_list:
        if filename.endswith("." + extension):
            full_filename = base_folder + filename
            file_list_filtered.append(full_filename)
    
    return sorted(file_list_filtered)


def get_spectrogram_array(file_list, n_mels, n_frames, spectrogram_folder_name, artist, from_file_list=False):

    spec_counter = len(file_list) 
    mel_spectrogram_array = np.empty((spec_counter,n_mels,n_frames)) 

    for image_index in range(spec_counter):

        if not from_file_list:
            mel_img = plt.imread("{}{}_{:06d}.png".format(spectrogram_folder_name, artist, image_index))
        else:
            mel_img = plt.imread("{}{}_{}.png".format(spectrogram_folder_name, artist, file_list[image_index][-10:-4]))
        
        mel_img = mel_img[:,:,0]
        mel_img = np.flip(mel_img, axis=0)
        mel_img = 1-mel_img
        mel_spectrogram_array[image_index] = mel_img

        if (image_index % 50 == 49):
            print(image_index+1, "/", spec_counter, " spectrograms processed", sep='')
            clear_output(wait=True)

    return mel_spectrogram_array


def split_dataset(mel_spectrogram_array, percentage_train=0.85, max_elements=-1):

    x_train_index = int (percentage_train * (len(mel_spectrogram_array) if max_elements <= 0 else max_elements))
    x_test_index = int ((1-percentage_train) * (len(mel_spectrogram_array) if max_elements <= 0 else max_elements)) + 1

    x_train = mel_spectrogram_array[:x_train_index]
    x_train = np.reshape(x_train,(len(x_train),-1))

    x_test = mel_spectrogram_array[-x_test_index:]
    x_test = np.reshape(x_test,(len(x_test),-1))
    
    print("Train: [0",",",x_train_index,")",sep='')
    print(" Test: [",len(mel_spectrogram_array)-x_test_index,",",len(mel_spectrogram_array),")",sep='')
    
    return x_train, x_test


def spectrogram_to_audio(data, n_mels, n_frames, sr, n_fft, hop_length, fmin, fmax):
    
    mel = np.reshape(data,(n_mels,n_frames))
    mel = -(1-mel) * 80
    mel = librosa.db_to_power(mel)

    y = librosa.feature.inverse.mel_to_audio(mel, sr=sr,
                                              n_fft=n_fft, hop_length=hop_length, 
                                              window=scipy.signal.hamming,
                                              fmin=fmin, fmax=fmax)
    
    return mel, y