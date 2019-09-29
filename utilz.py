from __future__ import division
import scipy.io
import numpy as np
import torch
from scipy import misc
import matplotlib.pyplot as plt
import os
from scipy.ndimage.interpolation import map_coordinates
from scipy import interpolate as ipol


np.seterr(divide='ignore', invalid='ignore')



def load_files(filename): 
    
    if filename.endswith('.mat'):
        file = scipy.io.loadmat(filename)
        keys = file.keys()  
        if 'opus' in keys:
            return np.array(file['opus'])
        if 'opus_nnmf' in keys:
            return np.array(file['opus_nnmf'])
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
        else:
            print('unknown format')
    
    if filename.endswith('.png'):
        return misc.imread(filename)
    




def show_labels(image, labels, prediction, results_path, i):
    _, a, b, _ = np.where(labels != 0)
    e, f = np.where(prediction != 0)
    _, x, y, z = image.shape
    image = image[0,0,:,:] ### pick one spectrum just to show image+labels

    
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
    
    



def binary(o): 
    mean = torch.mean(o)
    bin_img = torch.where((o.cpu() > mean.cpu()), torch.tensor(1), torch.tensor(0))
    return bin_img
    


def norm(ar):
    ar -= np.min(ar, axis=0)
    ar /= np.ptp(ar, axis=0)
    return ar




### from Quicknat, preprocess.py
def estimate_weights_mfb(labels):
    class_weights = np.zeros_like(labels)
    unique, counts = np.unique(labels, return_counts=True)
    median_freq = np.median(counts)
    weights = np.zeros(len(unique))
    for i, label in enumerate(unique):
        class_weights += (median_freq / counts[i]) * np.array(labels == label)
        
        
#        print('unique', unique)
#        print('counts', counts)
#        print('label: ', label)
#        print('i', i)
        
        weights[int(label)] = median_freq / counts[i]
#        
#        print('median_freq: ', median_freq)
#        print('counts[i]: ', counts[i])
#        print('weights: ', weights)
#        print('------------------------')
        
    grads = np.gradient(labels)
    edge_weights = (grads[0] ** 2 + grads[1] ** 2) > 0
    class_weights += 2 * edge_weights
    return class_weights, weights




# deform function
def elastic_deformation(image, x_coord, y_coord, dx, dy):
    """ Applies random elastic deformation to the input image 
        with given coordinates and displacement values of deformation points.
        Keeps the edge of the image steady by adding a few frame points that get displacement value zero.
    Input: image: array of shape (N.M,C) (Haven't tried it out for N != M), C number of channels
           x_coord: array of shape (L,) contains the x coordinates for the deformation points
           y_coord: array of shape (L,) contains the y coordinates for the deformation points
           dx: array of shape (L,) contains the displacement values in x direction
           dy: array of shape (L,) contains the displacement values in x direction
    Output: the deformed image (shape (N,M,C))
    """
        
    ## Preliminaries    
    # dimensions of the input image
    shape = image.shape
        
    # centers of x and y axis
    x_center = shape[1]/2
    y_center = shape[0]/2
    
    ## Construction of the coarse grid
    # deformation points: coordinates
            
    # anker points: coordinates    
    x_coord_anker_points = np.array([0, x_center, shape[1] - 1, 0, shape[1] - 1, 0, x_center, shape[1] - 1])
    y_coord_anker_points = np.array([0, 0, 0, y_center, y_center, shape[0] - 1, shape[0] - 1, shape[0] - 1])
    # anker points: values
    dx_anker_points = np.zeros(8)
    dy_anker_points = np.zeros(8)
    
    # combine deformation and anker points to coarse grid 
    x_coord_coarse = np.append(x_coord, x_coord_anker_points)
    y_coord_coarse = np.append(y_coord, y_coord_anker_points)
    coord_coarse = np.array(list(zip(x_coord_coarse, y_coord_coarse)))
    
    dx_coarse = np.append(dx, dx_anker_points)
    dy_coarse = np.append(dy, dy_anker_points)
        
    ## Interpolation onto fine grid
    # coordinates of fine grid
    coord_fine = [[x,y] for x in range(shape[1]) for y in range(shape[0])]
    # interpolate displacement in both x and y direction
    dx_fine = ipol.griddata(coord_coarse, dx_coarse, coord_fine, method = 'cubic') # cubic works better but takes longer
    dy_fine = ipol.griddata(coord_coarse, dy_coarse, coord_fine, method = 'cubic') # other options: 'linear'
    # get the displacements into shape of the input image (the same values in each channel)
    
    
    dx_fine = dx_fine.reshape(shape[0:2])
    dx_fine = np.stack([dx_fine]*shape[2], axis = -1)
    dy_fine = dy_fine.reshape(shape[0:2])
    dy_fine = np.stack([dy_fine]*shape[2], axis = -1)
    
    ## Deforming the image: apply the displacement grid
    # base grid 
    x, y, z = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]))
    # add displacement to base grid (-> new coordinates)
    indices = np.reshape(y+dy_fine, (-1, 1)), np.reshape(x+dx_fine, (-1, 1)), np.reshape(z, (-1, 1))
    # evaluate the image at the new coordinates
    deformed_image = map_coordinates(image, indices, order=2, mode='nearest')
    deformed_image = deformed_image.reshape(image.shape)
    
    return deformed_image




        