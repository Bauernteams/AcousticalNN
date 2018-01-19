'''
Created on 15.01.2018

@author: nikki
'''
import numpy as np
import glob
import os
import librosa
import matplotlib.pyplot as plt
from matplotlib.pyplot import specgram
import tensorflow as tf

def load_sound_files(file_paths):
    raw_sounds = []
    for fp in file_paths:
        X,sr = librosa.load(fp)
        raw_sounds.append(X)
    return raw_sounds

def parse_audio_files(parent_dir,sub_dirs,file_ext='*.wav', maxDataPerClass=False):
    features, labels = np.empty((0,193)), np.empty(0)
    
    #NB: limit the data input to maxDataPerClass
    if maxDataPerClass:
        dataLimiter = maxDataPerClass * np.ones((2,1),dtype=np.int8)
    else:
        dataLimiter = np.inf * np.ones((2,1), dtype=np.int8)
    print("Starting to read data files...")
    for sub_dir in sub_dirs:
        for fn in glob.glob(os.path.join(parent_dir, sub_dir, file_ext)):
            if os.path.getsize(fn) <= 25000:
                break
            label = np.int(fn.split('/')[2].split('-')[1])
            if dataLimiter[label] > 0:
                print(fn.split('/')[2])
                dataLimiter[label] -= 1
                mfccs, chroma, mel, contrast,tonnetz = extract_feature(fn)
                ext_features = np.hstack([mfccs,chroma,mel,contrast,tonnetz])
                features = np.vstack([features,ext_features])
                labels = np.append(labels, label)
            
            print("Missing data per class:", np.transpose(dataLimiter))
            if not dataLimiter.any():
                print("finished?")
                break
        if maxDataPerClass:
            print("Missing data per class:", np.transpose(dataLimiter)) 
    return np.array(features), np.array(labels, dtype = np.int)

def parse_audio_file(filePath):
    label = np.int(filePath.split('/')[-1].split('-')[1])
    mfccs, chroma, mel, contrast,tonnetz = extract_feature(filePath)
    features = np.hstack([mfccs,chroma,mel,contrast,tonnetz])
            
    return np.array(features), np.array(label, dtype = np.int)

def extract_feature(file_name):
    X, sample_rate = librosa.load(file_name)
    stft = np.abs(librosa.stft(X))
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T,axis=0)
    return mfccs,chroma,mel,contrast,tonnetz

def one_hot_encode(labels):
    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels))
    one_hot_encode = np.zeros((n_labels,n_unique_labels))
    one_hot_encode[np.arange(n_labels), labels] = 1
    return one_hot_encode

def use_neural_network(input_data, n_dim=193, n_classes=2, n_hidden_units_one=280, n_hidden_units_two=300, sd=None, checkpoint_path=None):
    if not sd:
        sd=1/np.sqrt(n_dim)
    X = tf.placeholder(tf.float32,[None,n_dim])
    Y = tf.placeholder(tf.float32,[None,n_classes])
    hidden_1_layer = {'f_fum': n_hidden_units_one,
                      'weight':tf.Variable(tf.random_normal([n_dim,n_hidden_units_one], mean = 0, stddev=sd)),
                      'bias':  tf.Variable(tf.random_normal([n_hidden_units_one], mean = 0, stddev=sd))}
    hidden_2_layer = {'f_fum': n_hidden_units_two,
                      'weight':tf.Variable(tf.random_normal([n_hidden_units_one,n_hidden_units_two], mean = 0, stddev=sd)),
                      'bias':  tf.Variable(tf.random_normal([n_hidden_units_two], mean = 0, stddev=sd))}
    output_layer = {'f_fum': None,
                    'weight':tf.Variable(tf.random_normal([n_hidden_units_two,n_classes], mean = 0, stddev=sd)),
                    'bias':  tf.Variable(tf.random_normal([n_classes], mean = 0, stddev=sd)),}
    saver = tf.train.Saver()
    h_1 = tf.nn.tanh(tf.matmul(X,hidden_1_layer["weight"]) + hidden_1_layer["bias"])
    h_2 = tf.nn.sigmoid(tf.matmul(h_1,hidden_2_layer["weight"]) + hidden_2_layer["bias"])
    prediction = tf.nn.softmax(tf.matmul(h_2,output_layer['weight']) + output_layer['bias'])
    
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        if checkpoint_path:
            ckpt = checkpoint_path
        else:
            ckpt = "/home/nikki/Documents/LiClipse Workspace/Road Accoustics/model.ckpt"
        saver.restore(sess,ckpt)
        
        features, label = parse_audio_file(input_data)
        evalValue = prediction.eval(feed_dict={X:[features]})
        #print("evalValue: ", max(evalValue[0]))
        result = (sess.run(tf.argmax(prediction.eval(feed_dict={X:[features]}),1)))
        sound_names = ["dry","wet"]
        print("\nResult:\t\t", int(result), sound_names[int(result)], "\nOriginal Label:\t", label, sound_names[int(label)])
        if(result == label):
            print("CORRECT! :)")
            return True, evalValue
        else:
            print("False... :(")
            return False, evalValue

def import_test_data():
    return parse_audio_files("Sound-Data",["test"])
    
########
   

########
#PLOTS

def plot_waves(sound_names,raw_sounds):
    i = 1
    fig = plt.figure(figsize=(25,60))
    for n,f in zip(sound_names,raw_sounds):
        plt.subplot(10,1,i)
        librosa.display.waveplot(np.array(f),sr=22050)
        plt.title(n.title())
        i += 1
    
    plt.suptitle('Figure 1: Waveplot',x=0.5, y=0.915,fontsize=18)
    plt.show()
    
def plot_specgram(sound_names,raw_sounds):
    i = 1
    fig = plt.figure(figsize=(25,60))
    for n,f in zip(sound_names,raw_sounds):
        plt.subplot(10,1,i)
        specgram(np.array(f), Fs=22050)
        plt.title(n.title())
        i += 1
    plt.suptitle('Figure 2: Spectrogram',x=0.5, y=0.915,fontsize=18)
    plt.show()

def plot_log_power_specgram(sound_names,raw_sounds):
    i = 1
    fig = plt.figure(figsize=(25,60))
    for n,f in zip(sound_names,raw_sounds):
        plt.subplot(10,1,i)
        D = librosa.logamplitude(np.abs(librosa.stft(f))**2, ref_power=np.max)
        librosa.display.specshow(D,x_axis='time' ,y_axis='log')
        plt.title(n.title())
        i += 1
    plt.suptitle('Figure 3: Log power spectrogram',x=0.5, y=0.915,fontsize=18)
    plt.show()