import scipy.io
import numpy as np
import torch
from scipy import misc
import matplotlib.pyplot as plt
import os

np.seterr(divide='ignore', invalid='ignore')



def load_files(filename): 
    
    if filename.endswith('.mat'):
        file = scipy.io.loadmat(filename)
        keys = file.keys()  
        if 'Recons' in keys:
            return np.array(file['Recons'])
        if 'rec_img_nnReg' in keys:            
            return np.array(file['rec_img_nnReg'])
        if 'us_enhanced' in keys:
            return np.array(file['us_enhanced'])
        if 'ROI' in keys:
            return np.array(file['ROI'])
        if 'US' in keys:
            return np.array(file['US'])
    
    if filename.endswith('.png'):
        return misc.imread(filename)
    
    else:
        print('unknown format')



def show_labels(image, labels, prediction, results_path, i):
    
    _, a, b = np.where(labels != 0)
    e, f = np.where(prediction != 0)
    _, x, y, z = image.shape
    image = image[0,-1,:,:] ### pick one spectrum just to show image+labels

    
    plt.figure()
    plt.imshow(image, cmap='gray')
    aa = plt.scatter(b, a, s=1, marker='o', c='red', alpha=0.5)
    aa.set_label('label')
    plt.legend()
    bb = plt.scatter(f, e, s=2, marker='o', c='blue', alpha=0.1)
    bb.set_label('prediction')
    plt.legend()
    plt.axis('off')
    plt.savefig(os.path.join(results_path, 'label plus pred' + str(i) + '.png'))
    
    
    
def median_frequency_balancing(labels):
    
    Labels = np.unique(labels)
    NumClass = len(Labels)
    ClassWeights = np.zeros((NumClass))
    ClassFreq = np.zeros((1, NumClass))
    
    # Estimate Class Frequency
    for i in range(NumClass):
        #l = labels
        pixel_in_this_class, test = np.where(labels.astype(int) == i)
        ClassFreq[0, i] = len(pixel_in_this_class) / len(labels)
    
    MedianFreq = np.median(ClassFreq)
    
    for j in range(NumClass):
        ClassWeights[j] = MedianFreq / ClassFreq[0,j]        
    return ClassWeights



def binar(o): 
    
    mean = torch.mean(o)
    bin_img = torch.where((o.cpu() > mean.cpu()), torch.tensor(1), torch.tensor(0))
    return bin_img
    


def norm(ar):
    
    ar -= np.min(ar, axis=0)
    ar /= np.ptp(ar, axis=0)
    return ar





        