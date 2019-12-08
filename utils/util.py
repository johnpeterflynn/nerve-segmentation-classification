from __future__ import division

import collections
import json
import os
import pickle
from argparse import Action
from enum import Enum
from itertools import repeat
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io
import torch
import torch.nn as nn
import torch.nn.functional as F
from polyaxon_client.tracking import (Experiment, get_data_paths,
                                      get_outputs_path)
from scipy import interpolate as ipol
from scipy import misc
from scipy.ndimage.interpolation import map_coordinates
from torch.autograd import Variable

from utils import visualization

np.seterr(divide='ignore', invalid='ignore')


class RuntimeEnvironment(Enum):
    LOCAL = "local"
    COLAB = "colab"
    POLYAXON = "polyaxon"

    def __str__(self):
        return self.value


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=collections.OrderedDict)


def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader


class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(
            index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / \
            self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)


def load_pickle_file(dataset_location):
    max_bytes = 2 ** 31 - 1
    data = {}
    new_data = None
    print("Loading file", dataset_location)
    bytes_in = bytearray(0)
    input_size = os.path.getsize(dataset_location)
    with open(dataset_location, 'rb') as f_in:
        for _ in range(0, input_size, max_bytes):
            bytes_in += f_in.read(max_bytes)
    new_data = pickle.loads(bytes_in)
    data.update(new_data)

    del new_data
    return data


def namedtuple_with_defaults(typename, field_names, default_values=()):
    T = collections.namedtuple(typename, field_names)
    T.__new__.__defaults__ = (None,) * len(T._fields)
    if isinstance(default_values, collections.Mapping):
        prototype = T(**default_values)
    else:
        prototype = T(*default_values)
    T.__new__.__defaults__ = tuple(prototype)
    return T


class EnvironmentAction(Action):

    def __call__(self, parser, namespace, value, option_string=None):

        if value == RuntimeEnvironment.POLYAXON:
            save_dir = get_outputs_path()
            setattr(namespace, "save_dir", save_dir)

            # TODO: We can do this programmaticlly
            # But we need to agree upon a certain format
            # for now, just pass it as an argument through
            # cli
            #data_dir = get_data_paths()
            #setattr(namespace, "data_dir", data_dir['data1'])


def truncated_normal_(tensor, mean=0, std=1):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)


def init_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        truncated_normal_(m.bias, mean=0, std=0.001)


def init_weights_orthogonal_normal(m):
    if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        nn.init.orthogonal_(m.weight)
        truncated_normal_(m.bias, mean=0, std=0.001)


def l2_regularisation(m):
    l2_reg = None

    for W in m.parameters():
        if l2_reg is None:
            l2_reg = W.norm(2)
        else:
            l2_reg = l2_reg + W.norm(2)
    return l2_reg


def save_mask_prediction_example(mask, pred, iter):
    plt.imshow(pred[0, :, :], cmap='Greys')
    plt.savefig('images/'+str(iter)+"_prediction.png")
    plt.imshow(mask[0, :, :], cmap='Greys')
    plt.savefig('images/'+str(iter)+"_mask.png")


def norm(ar):
    ar -= np.min(ar, axis=0)
    ar /= np.ptp(ar, axis=0)
    return ar


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

    # Preliminaries
    # dimensions of the input image
    shape = image.shape

    # centers of x and y axis
    x_center = shape[1]/2
    y_center = shape[0]/2

    # Construction of the coarse grid
    # deformation points: coordinates

    # anker points: coordinates
    x_coord_anker_points = np.array(
        [0, x_center, shape[1] - 1, 0, shape[1] - 1, 0, x_center, shape[1] - 1])
    y_coord_anker_points = np.array(
        [0, 0, 0, y_center, y_center, shape[0] - 1, shape[0] - 1, shape[0] - 1])
    # anker points: values
    dx_anker_points = np.zeros(8)
    dy_anker_points = np.zeros(8)

    # combine deformation and anker points to coarse grid
    x_coord_coarse = np.append(x_coord, x_coord_anker_points)
    y_coord_coarse = np.append(y_coord, y_coord_anker_points)
    coord_coarse = np.array(list(zip(x_coord_coarse, y_coord_coarse)))

    dx_coarse = np.append(dx, dx_anker_points)
    dy_coarse = np.append(dy, dy_anker_points)

    # Interpolation onto fine grid
    # coordinates of fine grid
    coord_fine = [[x, y] for x in range(shape[1]) for y in range(shape[0])]
    # interpolate displacement in both x and y direction
    # cubic works better but takes longer
    dx_fine = ipol.griddata(coord_coarse, dx_coarse,
                            coord_fine, method='cubic')
    # other options: 'linear'
    dy_fine = ipol.griddata(coord_coarse, dy_coarse,
                            coord_fine, method='cubic')
    # get the displacements into shape of the input image (the same values in each channel)

    dx_fine = dx_fine.reshape(shape[0:2])
    dx_fine = np.stack([dx_fine]*shape[2], axis=-1)
    dy_fine = dy_fine.reshape(shape[0:2])
    dy_fine = np.stack([dy_fine]*shape[2], axis=-1)

    # Deforming the image: apply the displacement grid
    # base grid
    x, y, z = np.meshgrid(np.arange(shape[0]), np.arange(
        shape[1]), np.arange(shape[2]))
    # add displacement to base grid (-> new coordinates)
    indices = np.reshape(y+dy_fine, (-1, 1)), np.reshape(x +
                                                         dx_fine, (-1, 1)), np.reshape(z, (-1, 1))
    # evaluate the image at the new coordinates
    deformed_image = map_coordinates(image, indices, order=2, mode='nearest')
    deformed_image = deformed_image.reshape(image.shape)

    return deformed_image


def binary(o):
    mean = torch.mean(o)
    bin_img = torch.where((o.cpu() > mean.cpu()),
                          torch.tensor(1), torch.tensor(0))
    return bin_img


# from Quicknat, preprocess.py
def estimate_weights_mfb(labels):
    class_weights = np.zeros_like(labels)
    unique, counts = np.unique(labels, return_counts=True)
    median_freq = np.median(counts)
    weights = np.zeros(len(unique))
    for i, label in enumerate(unique):
        class_weights += (median_freq / counts[i]) * np.array(labels == label)

        weights[int(label)] = median_freq / counts[i]

    grads = np.gradient(labels)
    edge_weights = (grads[0] ** 2 + grads[1] ** 2) > 0
    class_weights += 2 * edge_weights
    return class_weights, weights


def show_labels(image, labels, prediction, results_path, i):
    _, a, b, _ = np.where(labels != 0)
    e, f = np.where(prediction != 0)
    _, x, y, z = image.shape
    image = image[0, 0, :, :]  # pick one spectrum just to show image+labels

    plt.figure()
    plt.imshow(image, cmap='gray')
    aa = plt.scatter(b, a, s=1, marker='o', c='red', alpha=0.5)
    aa.set_label('label')
    plt.legend()
    bb = plt.scatter(f, e, s=2, marker='o', c='blue', alpha=0.1)
    bb.set_label('prediction')
    plt.legend()
    plt.axis('off')
    plt.savefig(os.path.join(
        results_path, 'label plus pred' + str(i) + '.png'))


def impose_labels_on_image(image, labels, prediction):
    a, b = np.where(labels != 0)
    e, f = np.where(prediction != 0)

    fig = plt.figure(figsize=(4, 4))
    plt.imshow(image, cmap='gray')
    aa = plt.scatter(b, a, s=1, marker='o', c='red', alpha=0.5)
    # aa.set_label('label')
    # plt.legend()
    bb = plt.scatter(f, e, s=2, marker='o', c='blue', alpha=0.1)
    # bb.set_label('prediction')
    # plt.legend()
    plt.axis('off')
    plt.tight_layout(0)

    fig.canvas.draw()

    # Convert from figure to image
    buf = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    plt.close(fig)

    # Convert to pytorch tensor with format C x H x W
    torch_buf = torch.from_numpy(buf)
    torch_buf = torch_buf.permute(2, 0, 1)
    torch_buf = torch_buf.unsqueeze(0)

    return torch_buf


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


def build_segmentation_grid(metrics_sample_count, targets, inputs, samples, avg_output):
    """
    TODO: Make more generic, currently it works only for binary segmentation
        inputs: [BATCH_SIZE x NUM_CHANNELS x H x W] #  NUM_CHANNELS = 7 for opus 
        samples: [BATCH_SIZE x SAMPLE_SIZE x NUM_CHANNELS x H x W]
        targets: [BATCH_SIZE x  H x W]
    """
    gt_title = ['Input Image', 'GT Segmentation']
    #gt_title = ['GT Segmentation']
    s_titles = [f'S_{i}' for i in range(metrics_sample_count)]
    titles = gt_title + s_titles + ['Avg-Output'] + ['Variance']

    heatmaps = visualization.samples_heatmap(samples)

    # add num of channels dim - needed for the metric format
    target = targets.unsqueeze(1).unsqueeze(1)
    avg_output = avg_output.unsqueeze(1)

    inputs = inputs[:, 0, :, :]  # pick one spectrum just to show image+labels
    inputs = inputs.unsqueeze(1).unsqueeze(1)

    overlayed_labels = torch.cat((inputs, target), dim=1)
    vis_data = torch.cat((overlayed_labels, samples), dim=1)
    vis_data = torch.cat((vis_data, avg_output), dim=1)

    img_metric_grid = visualization.make_image_metric_grid(vis_data,
                                                           enable_helper_dots=True,
                                                           titles=titles,
                                                           heatMaps=heatmaps)
    return img_metric_grid


def save_grid(grid, save_dir, idx=None):

    plt.figure(figsize=(100, 100))
    grid = grid.permute(1, 2, 0)
    plt.imshow(grid)
    save_dir = Path(save_dir) / 'test-images/'
    save_dir.mkdir(parents=True, exist_ok=True)
    plt.axis("off")

    if idx is None:
        plt.savefig(save_dir / ("test-result.png"))
    else:
        plt.savefig(save_dir / (str(idx) + "test-result.png"))

def save_img(img, save_dir, idx):

    plt.figure(figsize=(50, 50))
    plt.imshow(img)
    save_dir = Path(save_dir) / 'test-images/'
    save_dir.mkdir(parents=True, exist_ok=True)
    plt.axis("off")

    plt.savefig(save_dir / (str(idx) + ".png"))




def argmax_over_dim(samples, dim=2, keepdim=True):
    _, idx = torch.max(samples, dim=dim)
    if keepdim:
        idx.unsqueeze_(dim=dim)
    return idx.float()


def sample_and_compute_mean(model, data, num_samples, num_channels_model, device):
    """
        Samples the model 'num_samples' times
        then computes the average of these samples
        which would be the ouput of the model. And the 
        samples will be used later to compute uncertainty
        metrics.

        model: the model to evaluate the data with
        data: [BATCH_SIZE x C x H x W]
        num_samples: Number of MC samples
        num_channels_model: the number of the channels in the model output
        device: device to use (cuda or cpu)
    """

    batch_size, num_channels, image_size = data.shape[0], num_channels_model, tuple(data.shape[2:])
    samples = torch.zeros(
        (batch_size, num_samples, num_channels, *image_size)).to(device)
    for i in range(num_samples):
        samples[:, i, ...] = model(data)

    return samples.mean(dim=1), samples
