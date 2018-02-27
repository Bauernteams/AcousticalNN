import os
import librosa


inpath = "Q:/Repositories/Samples/Sorted/RaodType/"
printevery = 1

def get_class_names(path="Samples/"):  # class names are subdirectory names in Samples/ directory
    class_names = os.listdir(path)
    return class_names


class_names = get_class_names(path=inpath)   # get the names of the subdirectories
for idx, classname in enumerate(class_names):
    class_files = os.listdir(inpath + classname)  
    for idx2, infilename in enumerate(class_files):
        audio_path_link = inpath + classname + '/' + infilename

        shell = win32com.client.Dispatch("WScript.Shell")
        shortcut = shell.CreateShortCut(audio_path_link)
        audio_path = shortcut.Targetpath
        
        #start = timer()
        fileSize = os.path.getsize(audio_path)
        if fileSize <= 0:
            print("File smaller than 350kb (",fileSize,") : skipping ", audio_path)
            continue
        aud, sr = librosa.load(audio_path, sr=None)
        print(aud)