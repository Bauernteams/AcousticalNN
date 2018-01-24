'''
Created on 15.01.2018

@author: nikki
'''
import numpy as np
import glob
import os
import librosa
import tensorflow as tf

def load_sound_files(file_paths):
    raw_sounds = []
    for fp in file_paths:
        X,sr = librosa.load(fp,44100)
        raw_sounds.append(X)
    return raw_sounds

def parse_audio_files(parent_dir,sub_dirs, file_ext='*.wav', label_count=2, maxDataPerClass=False, splitSound_ms = None):
    labels = np.empty(0)
    #NB: limit the data input to maxDataPerClass
    if maxDataPerClass:
        dataLimiter = maxDataPerClass * np.ones((label_count,1),dtype=np.int8)
    else:
        dataLimiter = np.inf * np.ones((label_count,1), dtype=np.int8)
    print("Starting to read data files...")
    features = None
    for sub_dir in sub_dirs:
        for fn in glob.glob(os.path.join(parent_dir, sub_dir, file_ext)):
            if os.path.getsize(fn) <= 25000:
                break
            label = np.int(fn.split('/')[2].split('-')[1])
            if dataLimiter[label] > 0:
                print(fn.split('/')[2])
                dataLimiter[label] -= 1
                feature_list = extract_feature(fn, splitSound_ms)
                ext_features = np.hstack(feature_list)
                #print("feature_list:", feature_list.shape, "\nlen(feature_list)", len(feature_list))
                #print("ext_features:", ext_features, "\nlen(ext_features)", len(ext_features))
                if features is None:
                    features = np.empty((0,ext_features.shape[-1]))                
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
    features = extract_feature(filePath)
    print("Shape: ", features.shape())
    features_list = np.hstack([*features])
            
    return np.array(features_list), np.array(label, dtype = np.int)

def extract_feature(file_name, splitSound_ms):
    X, sample_rate = librosa.load(file_name,44100)
    
    stft = np.abs(librosa.stft(X))
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T,axis=0)
    #print("len:\n stft:", len(stft), "\nmfccs:", len(mfccs), "\nchroma:", len(chroma), "\nmel:", len(mel), "\ncontrast:", len(contrast), "\ntonnetz:", len(tonnetz))
    #return stft
    return mfccs, chroma, mel, contrast, tonnetz

def one_hot_encode(labels):
    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels))
    one_hot_encode = np.zeros((n_labels,n_unique_labels))
    one_hot_encode[np.arange(n_labels), labels] = 1
    return one_hot_encode

from sklearn.metrics import precision_recall_fscore_support
def trainNN(features, labels, splitTestTrain_rel = 0.70, learning_rate = 0.01, training_epochs = 50, hiddenLayerNeuron_count=[280,300], LayerActivationFunctions=[0,1,2], sd=None, checkpoint_path=None):
    if sd is None:
        sd = 1/np.sqrt(features.shape[1])
    lAllLayerNeuron_count = [features.shape[1], *hiddenLayerNeuron_count, len(set(labels))]
    X = tf.placeholder(tf.float32,[None,features.shape[1]])
    Y = tf.placeholder(tf.float32,[None,len(set(labels))])
    if checkpoint_path is None:
        checkpoint_path = os.path.join(os.getcwd(), 'model.ckpt')
    
    Layer = []
    h = [X]
    for j in range(len(lAllLayerNeuron_count))[:-1]:
        Layer.append({'f_fum': lAllLayerNeuron_count[j+1],
                      'weight':tf.Variable(tf.random_normal([lAllLayerNeuron_count[j],lAllLayerNeuron_count[j+1]], mean = 0, stddev=sd)),
                      'bias':  tf.Variable(tf.random_normal([lAllLayerNeuron_count[j+1]], mean = 0, stddev=sd))})
        
        if LayerActivationFunctions[j] == 0:
            h.append(tf.nn.tanh(tf.matmul(h[j],Layer[j]["weight"]) + Layer[j]["bias"]))
        if LayerActivationFunctions[j] == 1:
            h.append(tf.nn.sigmoid(tf.matmul(h[j],Layer[j]["weight"]) + Layer[j]["bias"]))
        if LayerActivationFunctions[j] == 2:
            h.append(tf.nn.softmax(tf.matmul(h[j],Layer[j]["weight"]) + Layer[j]["bias"]))   
        
    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()
    init = tf.initialize_all_variables()
                  
    labels = one_hot_encode(labels)
    
    train_test_split = np.random.rand(len(features)) < splitTestTrain_rel
    train_x = features[train_test_split]
    train_y = labels[train_test_split]
    test_x = features[~train_test_split]
    test_y = labels[~train_test_split]
        
    cost_function = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(h[-1]), reduction_indices=[1])) 
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)
    
    correct_prediction = tf.equal(tf.argmax(h[-1],1), tf.argmax(Y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    # In[ ]:
    
    print("8")
    cost_history = np.empty(shape=[1],dtype=float)
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(training_epochs):
            print("Epoche %d von %d" % (epoch,training_epochs))
            #if epoch != 0:
            #    saver.restore(sess,checkpoint_path)
            _,cost = sess.run([optimizer,cost_function],feed_dict={X:train_x,Y:train_y})
            cost_history = np.append(cost_history,cost)
            #savePath = saver.save(sess, checkpoint_path)
        
        y_pred = sess.run(tf.argmax(h[-1],1),feed_dict={X: test_x})
        y_true = sess.run(tf.argmax(test_y,1))
        # Save the variables to disk.
        savePath = saver.save(sess, checkpoint_path)
        print("saved checkpoint in: ", savePath)
        
    # In[15]:
    print("9")
    
    fig = plt.figure(figsize=(10,8))
    plt.plot(cost_history)
    plt.ylabel("Cost")
    plt.xlabel("Iterations")
    plt.axis([0,training_epochs,0,np.max(cost_history)])
    plt.show()
    
    p,r,f,s = precision_recall_fscore_support(y_true, y_pred, average='micro')
    print ("F-Score:", round(f,3))
    
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

def splitSoundData(sample, sample_rate, length_ms=100):
    lenSampleFull_ms = len(sample)/sample_rate * 1000
    splitSample_count = int(lenSampleFull_ms/length_ms)
    splitSampleList = []
    print("i: ", splitSample_count)
    for i in range(0, len(sample), splitSample_count):
        splitSampleList.append(sample[i:i + splitSample_count])
    return splitSampleList
    
########
   

########
import librosa.display
import matplotlib.pyplot as plt
from matplotlib.pyplot import specgram

#PLOTS
def plot_waves(sound_names,raw_sounds):
    i = 1
    fig = plt.figure(figsize=(25,60))
    for n,f in zip(sound_names,raw_sounds):
        plt.subplot(len(sound_names),1,i)
        librosa.display.waveplot(np.array(f),sr=44100)
        plt.title(n.title())
        i += 1
    
    plt.suptitle('Figure 1: Waveplot',x=0.5, y=0.915,fontsize=18)
    plt.show()
    
def plot_specgram(sound_names,raw_sounds):
    i = 1
    fig = plt.figure(figsize=(25,60))
    for n,f in zip(sound_names,raw_sounds):
        plt.subplot(len(sound_names),1,i)
        specgram(np.array(f), Fs=44100)
        plt.title(n.title())
        i += 1
    plt.suptitle('Figure 2: Spectrogram',x=0.5, y=0.915,fontsize=18)
    plt.show()

def plot_log_power_specgram(sound_names,raw_sounds):
    i = 1
    fig = plt.figure(figsize=(25,60))
    for n,f in zip(sound_names,raw_sounds):
        plt.subplot(len(sound_names),1,i)
        D = librosa.logamplitude(np.abs(librosa.stft(f))**2, ref_power=np.max)
        librosa.display.specshow(D,x_axis='time' ,y_axis='log')
        plt.title(n.title())
        i += 1
    plt.suptitle('Figure 3: Log power spectrogram',x=0.5, y=0.915,fontsize=18)
    plt.show()
    
def compareSpecgram(sound_names, raw_sounds):
    i = 1
    for n,f in zip(sound_names,raw_sounds):
        plt.subplot(len(sound_names),1,i)
        specgram(np.array(f), Fs=44100)
        plt.title(n.title())
        i += 1
    plt.suptitle('Figure 2: Spectrogram',x=0.5, y=0.915,fontsize=18)
    plt.show()
    
def compareLogPowerSpecgram(sound_names, raw_sounds):
    i = 1
    for n,f in zip(sound_names,raw_sounds):
        plt.subplot(len(sound_names),2,2*i-1)
        D = librosa.logamplitude(np.abs(librosa.stft(f,2*4096))**2, ref_power=np.max)
        librosa.display.specshow(D,x_axis='time' ,y_axis='log', sr=44100)
        plt.title(n.title())
        plt.subplot(len(sound_names),2,2*i)
        E = librosa.logamplitude(np.abs(librosa.stft(f, 4*4096))**2, ref_power=np.max)
        librosa.display.specshow(E,x_axis='time' ,y_axis='log', sr=44100)
        plt.title(n.title())
        i += 1
    plt.suptitle('Figure 3: Log power spectrogram',x=0.5, y=0.915,fontsize=18)
    plt.show()