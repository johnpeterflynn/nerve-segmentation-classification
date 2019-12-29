import torch
from model.loss import dice as dice_loss
import numpy as np
from itertools import combinations

_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)


def dice_score(output, target):
    return 1 - dice_loss(output, target.long())


def ged(samples, targets):
    """
        Generalized Energy Distance function from Probabilistic UNet and PhiSeg

        PhiSeg paper [page 6]: https://arxiv.org/pdf/1906.04045.pdf
        Probabilistic UNet paper [page 4]: https://arxiv.org/abs/1806.05034

        :param samples: Tensor of shape [BATCH_SIZE x SAMPLE_SIZE x NUM_CHANNELS x H x W]
        :param targets: Tensor of shape [BATCH_SIZE x NUM_ANNOTATIONS x NUM_CHANNELS x H x W]
        :return: Scalar Tensor
    """
    batch_size = targets.shape[0]
    m = targets.shape[1]
    n = samples.shape[1]

    # Average GED between samples and ground truths
    first_term = torch.zeros(batch_size, dtype=torch.float)
    for sample_i in range(n):
        for gt_j in range(m):
            first_term += _ged_dist_func(samples[:, sample_i, ...], targets[:, gt_j, ...])
    first_term = (2 / (n * m)) * first_term.float()

    # Average GED between samples
    second_term = torch.zeros(batch_size, dtype=torch.float)
    for sample_i in range(n):
        for sample_j in range(n):
            second_term = _ged_dist_func(samples[:, sample_i, ...], samples[:, sample_j, ...])
    second_term = (1 / (n ** 2)) * second_term.float()

    # Average GED between ground truths
    third_term = torch.zeros(batch_size, dtype=torch.float)
    for gt_i in range(m):
        for gt_j in range(m):
            third_term += _ged_dist_func(targets[:, gt_i, ...], targets[:, gt_j, ...])
    third_term = (1 / (m ** 2)) * third_term.float()

    geds = first_term - second_term - third_term

    return geds.mean().cpu().item()


def dice_agreement_in_samples(samples, _=None):
    """
        Average Dice Score over generated samples, Type 2 Uncertainty Measurement from Bayesian QuickNAT paper

        Bayesian QuickNAT paper [page 4-5]: https://arxiv.org/pdf/1811.09800.pdf
    :param samples: Tensor of shape [BATCH_SIZE x SAMPLE_SIZE x NUM_CHANNELS x H x W]
    :return: Scalar Tensor
    """
    batch_size = samples.shape[0]
    n = samples.shape[1]
    num_labels = samples.shape[2]
    dice_per_label = torch.zeros((batch_size, num_labels), dtype=torch.float)
    dice_per_label = dice_per_label.to(_device)

    num_pairs = 0
    for i, j in combinations(list(range(n)), 2):
        inp1 = samples[:, i, ...]
        inp2 = samples[:, j, ...]

        # Compute number of pixels intersecting
        intersection = inp1 * inp2
        sum_intersection = intersection.view(num_labels, -1).sum(1)
        # Compute number of pixels in union
        union = (inp1 + inp2)
        sum_union = (union > 0).view(num_labels, -1).sum(1)  # Will exclude double-intersection as well

        sum_i = sum_intersection[:].float()
        sum_u = sum_union[:].float()
        dice_per_label[:, :] += (sum_i + 1e-6) / (sum_u + 1e-6)

        num_pairs += 1

    dice_per_label[:, :] /= num_pairs
    return dice_per_label.mean().cpu().item()


def iou_samples_per_label(samples, _=None):
    """
        Intersection over Union between samples, Type 3 Uncertainty Measurement from Bayesian QuickNAT paper

        Bayesian QuickNAT paper [page 4-5]: https://arxiv.org/pdf/1811.09800.pdf

    :param samples: Tensor of shape [BATCH_SIZE x SAMPLE_SIZE x NUM_CHANNELS x H x W]
    :return: Scalar Tensor of shape
    """
    batch_size = samples.shape[0]
    n = samples.shape[1]
    num_labels = samples.shape[2]

    dice_per_label = torch.zeros((batch_size, num_labels)).float().to(_device)

    # Size of intersection and union is [NUM_CHANNELS x H x W]
    img_size = (batch_size, *samples.shape[2:])
    intersection = torch.ones(img_size).long().to(_device)
    union = torch.zeros(img_size).long().to(_device)

    # Intersection and union over all samples
    for i in range(n):
        intersection *= samples[:, i, ...].long()
        union += samples[:, i, ...].long()

    sum_intersection = intersection.view(num_labels, -1).sum(1)
    sum_union = (union > 0).view(num_labels, -1).sum(1)  # Will exclude double-intersection as well

    dice_per_label[:, ...] = (sum_intersection.float() + 1e-6) / (sum_union.float() + 1e-6)

    return dice_per_label.mean().cpu().item()


def pixel_wise_ce_samples(samples):
    """
        Average Pixel-wise Cross Entropy between Samples and their Mean.

        PhiSeg paper [page 6]: https://arxiv.org/pdf/1906.04045.pdf

    :param samples: Tensor of shape [NUM x C x H x W]
        where C is number of classes
    :return: Tensor of shape [H x W]
    """
    samples = samples.float()
    batch_size, N = samples.shape[0], samples.shape[1]

    mean_samples = samples.mean(1)

    gamma_maps = torch.zeros((batch_size, N, samples.shape[3], samples.shape[4]), dtype=torch.float).to(_device)
    for i in range(N):
        gamma_maps[:, i, ...] += _pixel_wise_xent(samples[:, i, ...], mean_samples)
    gamma_map = gamma_maps.mean(1)

    return gamma_map


def variance_ncc_samples(samples, g_truths):
    """
        Average Normalized Cross Correlation between "Pixel-wise Cross Entropy of Samples"
        and "Pixel-wise Cross Entropy between Samples & Annotators"

        PhiSeg paper [page 6]: https://arxiv.org/pdf/1906.04045.pdf

    :param samples: Tensor of shape [BATCH_SIZE x NUM_SAMPLES x NUM_CHANNELS x H x W]
        where C is number of classes
    :param g_truths: Tensor of shape [BATCH_SIZE x NUM_ANNOTATIONS x NUM_CHANNELS x H x W]
        where C is number of classes
    :return: Tensor of shape [H x W]
    """
    batch_size = samples.shape[0]
    N = samples.shape[1]
    M = g_truths.shape[1]

    batch_nccs = torch.zeros(batch_size).float().to(_device)
    batch_samples = samples[:, ...].float()
    batch_g_truths = g_truths[:, ...].float()

    gamma_map_ss = pixel_wise_ce_samples(batch_samples)

    E_sy_arr = torch.zeros((batch_size, M, N, batch_samples.size()[3], batch_samples.size()[4]))
    for j in range(M):
        for i in range(N):
            E_sy_arr[:, j, i, ...] = _pixel_wise_xent(batch_samples[:, i, ...], batch_g_truths[:, j, ...])

    E_sy = E_sy_arr.mean(dim=2)

    nccs = torch.zeros((batch_size, M)).float()
    for j in range(M):
        nccs[:, j] = _ncc(gamma_map_ss, E_sy[:, j, ...])

    batch_nccs[:] = nccs.mean(dim=1)

    return torch.mean(batch_nccs).cpu().item()


def _ged_dist_func(inp1: torch.Tensor, inp2: torch.Tensor):
    inp1 = inp1.float()
    inp2 = inp2.float()

    batch_size, num_labels = inp1.shape[0], inp1.shape[1]

    intersection = inp1 * inp2
    sum_intersection = intersection.view(batch_size, num_labels, -1).sum(2)
    union = (inp1 + inp2)
    sum_union = (union > 0).view(batch_size, num_labels, -1).sum(2)  # Will exclude double-intersection as well

    per_label_iou = torch.zeros(batch_size, num_labels).float()

    sum_i = sum_intersection[:]
    sum_u = sum_union[:]
    per_label_iou[:] = (sum_i.float() + 1e-6) / (sum_u + 1e-6)


    return 1 - per_label_iou.mean(dim=1)


def _pixel_wise_xent(sample, gt, eps=1e-8):
    log_sample = torch.log(sample + eps)
    return -1 * torch.sum(gt * log_sample, 1)


def _ncc(a,v, zero_norm=True, eps=1e-8):
    a = a.data.cpu().numpy()
    v = v.data.cpu().numpy()

    a = a.flatten()
    v = v.flatten()

    if zero_norm:

        a = (a - np.mean(a)) / (np.std(a) * len(a) + eps)
        v = (v - np.mean(v)) / (np.std(v) + eps)

    else:

        a = (a) / (np.std(a) * len(a) + eps)
        v = (v) / (np.std(v) + eps)

    return torch.from_numpy(np.correlate(a,v)).float()
