from __future__ import print_function

samplePath = "Q:/Repositories/Samples/Sorted/RaodType/"
outputPath = "Preproc/RoadType/"

''' 
Preprocess audio
'''
import numpy as np
import librosa
import librosa.display
import os
import sys
import win32com.client 

def get_class_names(path="Samples/"):  # class names are subdirectory names in Samples/ directory
    class_names = os.listdir(path)
    return class_names

def preprocess_dataset(inpath="Samples/", outpath="Preproc/"):

    if not os.path.exists(outpath):
        os.mkdir( outpath);   # make a new directory for preproc'd files

    class_names = get_class_names(path=inpath)   # get the names of the subdirectories
    nb_classes = len(class_names)
    print("class_names = ",class_names)
    for idx, classname in enumerate(class_names):   # go through the subdirs

        if not os.path.exists(outpath+classname):
            os.mkdir( outpath+classname);   # make a new subdirectory for preproc class

        class_files = os.listdir(inpath+classname)
        n_files = len(class_files)
        n_load = n_files
        print(' class name = {:14s} - {:3d}'.format(classname,idx),
            ", ",n_files," files in this class",sep="")

        printevery = 20
        for idx2, infilename in enumerate(class_files):
            audio_path_link = inpath + classname + '/' + infilename

            shell = win32com.client.Dispatch("WScript.Shell")
            shortcut = shell.CreateShortCut(audio_path_link)
            audio_path = shortcut.Targetpath

            if (0 == idx2 % printevery):
                print('\r Loading class: {:14s} ({:2d} of {:2d} classes)'.format(classname,idx+1,nb_classes),
                       ", file ",idx2+1," of ",n_load,": ",audio_path,sep="")
            #start = timer()
            fileSize = os.path.getsize(audio_path)
            if fileSize <= 40000:
                print("File smaller than 350kb (",fileSize,") : skipping ", audio_path)
                continue
            aud, sr = librosa.load(audio_path, sr=None)
            melgram = librosa.amplitude_to_db(librosa.feature.melspectrogram(aud, sr=sr, n_mels=96),ref=1.0)[np.newaxis,np.newaxis,:,:]
            outfile = outpath + classname + '/' + infilename+'.npy'
            np.save(outfile,melgram)

if __name__ == '__main__':
    preprocess_dataset(samplePath,outputPath)
                       

