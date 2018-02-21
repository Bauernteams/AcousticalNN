'''
Created on 15.01.2018

@author: nikki
'''
import numpy as np
import glob
import os
import librosa
import tensorflow as tf
def load_sound_files(file_paths, sampleRate):
    raw_sounds = []
    for fp in file_paths:
        X,sr = librosa.load(fp, sampleRate)
        raw_sounds.append(X)
    return raw_sounds

def parse_audio_files(parent_dir,sub_dirs, file_ext='*.wav', label_count=2, 
                      maxDataPerClass=False, splitSound_ms = None, sampleRate = 44100):
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
            print("fn:", fn)
            if fn.find("Run") != -1:
                print("parse_audio_file():", fn.split('/')[2].split(".")[0][3:6])
                actualRun = fn.split('/')[2].split(".")[0][3:6]
                label = getLabel(int(actualRun))
            else:
                label = np.int(fn.split('/')[2].split('-')[1])
            if dataLimiter[label] > 0:
                print(fn.split('/')[2])
                dataLimiter[label] -= 1
                featureSet_list = extract_feature(fn, splitSound_ms=splitSound_ms, sampleRate = sampleRate)
                if features is None:
                    features = np.empty((0,featureSet_list.shape[-1]))                
                features = np.vstack([features,np.vstack(featureSet_list)])
                print("features.shape:", features.shape)
                for i in range(featureSet_list.shape[0]):
                    labels = np.append(labels, label)
            
            print("Missing data per class:", np.transpose(dataLimiter))
            if not dataLimiter.any():
                print("finished?")
                break
        if maxDataPerClass:
            print("Missing data per class:", np.transpose(dataLimiter)) 
    return np.array(features), np.array(labels, dtype = np.int)

def parse_audio_file(filePath, splitSound_ms=None, sampleRate=44100):
    label = np.int(filePath.split('/')[-1].split('-')[1])
    features=None
    labels = np.empty(0)
      
    featureSet_list = extract_feature(filePath, splitSound_ms=splitSound_ms, sampleRate = sampleRate)
    for featureSet in featureSet_list:
        ext_features = np.hstack(featureSet)
        if features is None:
            features = np.empty((0,ext_features.shape[-1]))                
        features = np.vstack([features,ext_features])
        labels = np.append(labels, label)
            
    return np.array(features), np.array(labels, dtype = np.int)

def extract_feature2(file_name, splitSound_ms=None, sampleRate=44100):
    X,_ = librosa.load(file_name, sampleRate)
    #test=[X]
    #print("lWave:",len(lWave))
    #print("X1: ", len(X))
    featureSet_list = []
    for X in splitSoundData(X, sampleRate, splitSound_ms):
        #print("lWave: ", lWave)
        #print("X2: ", len(X))
        #print("n_fft=",len(X)*int(1000/splitSound_ms)-1)
        stft = np.abs(librosa.stft(X, n_fft=sampleRate-1))
        #print("stft: ",stft)
        print("Shape: ", stft.shape)
        #test.append(X)       
          
        #mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sampleRate, n_mfcc=40).T,axis=0)
        #chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sampleRate).T,axis=0)
        #mel = np.mean(librosa.feature.melspectrogram(X, sr=sampleRate).T,axis=0)
        #contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sampleRate).T,axis=0)
        #tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sampleRate).T,axis=0)
        #print("len:\n stft:", len(stft), "\nmfccs:", len(mfccs), "\nchroma:", len(chroma), "\nmel:", len(mel), "\ncontrast:", len(contrast), "\ntonnetz:", len(tonnetz))
        #return stft
        featureSet_list.append(np.hstack(stft))
    
    #compareSpecgram(["A","B","C","D"], test)
    
    return featureSet_list

def extract_feature(file_name, splitSound_ms=None, sampleRate=44100, featureType = "mean", freqWindow=None):
    X,_ = librosa.load(file_name, sampleRate)

    ## Feature extraction
    stft = np.abs(librosa.stft(X, n_fft=2048*2))
    feautureStack = np.empty((0,stft.shape[0]))
    #
    if featureType == "raw":
        for i in range(stft.shape[1]):
            feautureStack = np.vstack([feautureStack,list(zip(*stft))[i]])
    #        
    if featureType == "mean":
        feautureStack = np.vstack([np.mean(stft, 1)])
    ##
    
    ## Debugging
    #print("feautureStack: ",feautureStack)
    #print("feautureStack-Shape: ", feautureStack.shape)
    
    return feautureStack

def one_hot_encode(labels):
    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels))
    one_hot_encode = np.zeros((n_labels,n_unique_labels))
    one_hot_encode[np.arange(n_labels), labels] = 1
    return one_hot_encode

from sklearn.metrics import precision_recall_fscore_support
def trainNN(features, labels, splitTestTrain_rel = 0.70, learning_rate = 0.01, 
            training_epochs = 50, hiddenLayerNeuron_count=[280,300], 
            LayerActivationFunctions=[0,1,2], sd=None, checkpoint_path=None,
            checkpoint_name = None):
    #print("labels:",labels)
    if sd is None:
        sd = 1/np.sqrt(features.shape[1])
    lAllLayerNeuron_count = [features.shape[1], *hiddenLayerNeuron_count, len(set(labels))]
    X = tf.placeholder(tf.float32,[None,features.shape[1]])
    Y = tf.placeholder(tf.float32,[None,len(set(labels))])
    if checkpoint_name is None:
        checkpoint_name = 'lastSave.ckpt'
    if checkpoint_path is None:
        checkpoint_path = os.path.join(os.getcwd(),"Checkpoints", checkpoint_name)
    
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
    print("train_x.shape:",train_x.shape)
    train_y = labels[train_test_split]
    test_x = features[~train_test_split]
    print("test_x.shape:",test_x.shape)
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
    
def use_neural_network(inputData_path, n_classes=2, splitSound_ms=None, 
                       hiddenLayerNeuron_count=[280,300], 
                       LayerActivationFunctions=[0,1,2],
                       sd=None, checkpoint_name=None,
                       checkpoint_path=None,
                       sampleRate=44100):
    
    features, label = parse_audio_file(inputData_path, splitSound_ms=splitSound_ms, 
                                       sampleRate=sampleRate)
    print("features.shape:", features.shape)
    
    if sd is None:
        sd = 1/np.sqrt(features.shape[1])
    if checkpoint_name is None:
        checkpoint_name = "model.ckpt"
    if checkpoint_path is None:
        checkpoint_path = os.path.join(os.getcwd(),"Checkpoints", checkpoint_name)
        
    lAllLayerNeuron_count = [features.shape[1], *hiddenLayerNeuron_count, n_classes]
    X = tf.placeholder(tf.float32,[None,features.shape[1]])
    Y = tf.placeholder(tf.float32,[None,n_classes])
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
    
    with tf.Session() as sess:
        sess.run(init)
        saver.restore(sess,checkpoint_path)
        
        
        #y_pred = sess.run(tf.argmax(h[-1],1),feed_dict={X: features})
        
        for i in range(features.shape[0]):
            evalValue = h[-1].eval(feed_dict={X:[features[i]]})
            #print("evalValue: ", max(evalValue[0]))
            result = (sess.run(tf.argmax(h[-1].eval(feed_dict={X:[features[i]]}),1)))
            print("eval:", evalValue)
            print("result:", result)
        sound_names = ["dry","wet"]
        print("\nResult:\t\t", int(result), sound_names[int(result)], "\nOriginal Label:\t", label, sound_names[int(label)])
        if(result == label):
            print("CORRECT! :)", evalValue)
            return True, evalValue
        else:
            print("False... :(", evalValue)
            return False, evalValue

def import_test_data():
    return parse_audio_files("Sound-Data",["test"])

def splitSoundData(sample, sample_rate, length_ms = None):
    lenSampleFull_ms = int( len(sample)/sample_rate * 1000 )
    if length_ms is None:
        length_ms = lenSampleFull_ms
    splitSample_count = int(lenSampleFull_ms/length_ms)
    splitSampleList = []
    #print("Splitting ", lenSampleFull_ms, " long data in ", splitSample_count, "chunks!")
    for i in range(0, splitSample_count):
        #print("i:", i)
        #print("end:",(i+1)*int(len(sample)/splitSample_count))
        splitSampleList.append(sample[i*int(len(sample)/splitSample_count):(i+1)*int(len(sample)/splitSample_count)])
        #print(len(splitSampleList[-1]))
    return splitSampleList

def countSamples(parent_dir, sub_dirs, file_ext = "*.wav"):
    labelCount = [0,0]
    errornous = 0
    for sub_dir in sub_dirs:
        for fn in glob.glob(os.path.join(parent_dir, sub_dir, file_ext)):
            if os.path.getsize(fn) <= 25000:
                errornous+=1
                break
            label = np.int(fn.split('/')[2].split('-')[1])
            labelCount[label]+=1
    return labelCount, errornous

def getLabel(run):
    if run >= 0 and run < 45:
        return 0
    elif run >= 45 and run < 79:
        return 2
    elif run >= 79 and run < 157:
        return 1
    else:
        print("Inserted Sample is not known...!")
        return None

########
   

########
import librosa.display
import matplotlib.pyplot as plt
from matplotlib.pyplot import specgram

#PLOTS
def plot_waves(sound_names,raw_sounds, sampleRate=44100):
    i = 1
    fig = plt.figure(figsize=(25,60))
    for n,f in zip(sound_names,raw_sounds):
        plt.subplot(len(sound_names),1,i)
        librosa.display.waveplot(np.array(f),sr=sampleRate)
        plt.title(n.title())
        i += 1
    
    plt.suptitle('Figure 1: Waveplot',x=0.5, y=0.915,fontsize=18)
    plt.show()
    
def plot_specgram(sound_names,raw_sounds, sampleRate=44100):
    i = 1
    fig = plt.figure(figsize=(25,60))
    for n,f in zip(sound_names,raw_sounds):
        plt.subplot(len(sound_names),1,i)
        specgram(np.array(f),NFFT=sampleRate, Fs=sampleRate)
        plt.title(n.title())
        i += 1
    plt.suptitle('Figure 2: Spectrogram',x=0.5, y=0.915,fontsize=18)
    plt.show()

def plot_log_power_specgram(sound_names,raw_sounds, sampleRate=44100):
    i = 1
    fig = plt.figure(figsize=(25,60))
    for n,f in zip(sound_names,raw_sounds):
        plt.subplot(len(sound_names),1,i)
        D = librosa.logamplitude(np.abs(librosa.stft(f, n_fft=sampleRate))**2, ref_power=np.max)
        librosa.display.specshow(D,x_axis='time' ,y_axis='log')
        plt.title(n.title())
        i += 1
    plt.suptitle('Figure 3: Log power spectrogram',x=0.5, y=0.915,fontsize=18)
    plt.show()
    
def compareSpecgram(sound_names, raw_sounds,sampleRate = 44100):
    i = 1
    for n,f in zip(sound_names,raw_sounds):
        plt.subplot(len(sound_names),1,i)
        specgram(np.array(f), Fs=sampleRate)
        plt.title(n.title())
        i += 1
    plt.suptitle('Figure 2: Spectrogram',x=0.5, y=0.915,fontsize=18)
    plt.show()
    
def compareLogPowerSpecgram(sound_names, raw_sounds,sampleRate=44100):
    i = 1
    for n,f in zip(sound_names,raw_sounds):
        plt.subplot(len(sound_names),2,2*i-1)
        D = librosa.logamplitude(np.abs(librosa.stft(f,n_fft=sampleRate))**2, ref_power=np.max)
        librosa.display.specshow(D,x_axis='time' ,y_axis='log', sr=sampleRate)
        plt.title(n.title())
        plt.subplot(len(sound_names),2,2*i)
        E = librosa.logamplitude(np.abs(librosa.stft(f, n_fft=sampleRate))**2, ref_power=np.max)
        librosa.display.specshow(E,x_axis='time' ,y_axis='log', sr=sampleRate)
        plt.title(n.title())
        i += 1
    plt.suptitle('Figure 3: Log power spectrogram',x=0.5, y=0.915,fontsize=18)
    plt.show()
    
def comparePlot(first, second):
    plt.subplot(2,1,1)
    plt.plot(first)
    plt.subplot(2,1,2)
    plt.plot(second)
    plt.show()