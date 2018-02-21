'''
Created on 15.01.2018

@author: nikki
'''
import tools

#
## use trained neural network on new audio file:
#correctness, confidence = tools.use_neural_network("Sound-Data/test/wet14-1-2.wav", checkpoint_path="test/Ckpt/model.ckpt")
#print("correctness: \t", correctness, "\nconfidence: \t", confidence)

parent_dir = 'Sound-Data'
#sub_dirs = ['fold1','fold2','fold3']
sub_dirs = ['WABCO']
# Train new neural network
maxDataPerClass = 15 #776 is max Count for paper data
hiddenLayerNeuron_count = [150,100]
LayerActivationFunctions = [0,1,2]
training_epochs = 50
learning_rate = 0.01
splitSound_ms = 238
sampleRate = 22050
# 0: Learn; 1: Test                         
Programm = 3

checkpoint_name = "M-1-" + "-".join([str(maxDataPerClass), str(splitSound_ms), 
                                     str(sampleRate), str(training_epochs), 
                                     str(int(learning_rate*1000)), str(len(hiddenLayerNeuron_count)),
                                     *[str(i) for i in hiddenLayerNeuron_count]]) + ".ckpt"
            
print(checkpoint_name)

if Programm == 0:
    
    features, labels = tools.parse_audio_files(parent_dir,  sub_dirs, 
                                               maxDataPerClass  =maxDataPerClass, 
                                               splitSound_ms    =splitSound_ms, 
                                               sampleRate       =sampleRate)
    
    print("The amount of features in the ", len(labels), "samples is: ", features.shape[1])
    
    tools.trainNN(features, labels, 
                  hiddenLayerNeuron_count=hiddenLayerNeuron_count, 
                  LayerActivationFunctions=LayerActivationFunctions, 
                  learning_rate=learning_rate, 
                  training_epochs=training_epochs, checkpoint_name=checkpoint_name)
    
if Programm == 1:
    tools.use_neural_network("/home/nikki/Documents/LiClipse Workspace/Road Accoustics/Sound-Data/test/wet14-1-2.wav", 
                             splitSound_ms=splitSound_ms, 
                             hiddenLayerNeuron_count=hiddenLayerNeuron_count, 
                             LayerActivationFunctions=LayerActivationFunctions,
                             checkpoint_name=checkpoint_name,
                             sampleRate=sampleRate)
    
if Programm == 2:
    labelCount, err = tools.countSamples(parent_dir, sub_dirs)
    print("[Dry, Wet]:", labelCount, "\n errors:", err)
    
if Programm == 3:
    
    features, labels = tools.parse_audio_files(parent_dir,  sub_dirs, 
                                               maxDataPerClass  =maxDataPerClass, 
                                               splitSound_ms    =splitSound_ms, 
                                               sampleRate       =sampleRate)
    
    print("The amount of features in the ", len(labels), "samples is: ", features.shape[1])
    
    tools.trainNN(features, labels, 
                  hiddenLayerNeuron_count=hiddenLayerNeuron_count, 
                  LayerActivationFunctions=LayerActivationFunctions, 
                  learning_rate=learning_rate, 
                  training_epochs=training_epochs, checkpoint_name=checkpoint_name)