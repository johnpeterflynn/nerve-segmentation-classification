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

    geds = torch.zeros(batch_size).float()
    # TODO tensorize
    for b in range(batch_size):
        # Average GED between samples and ground truths
        first_term = torch.tensor(0, dtype=torch.float)
        for sample_i in range(n):
            for gt_j in range(m):
                first_term += _ged_dist_func(samples[b, sample_i, ...], targets[b, gt_j, ...])
        first_term = (2 / (n * m)) * first_term.float()

        # Average GED between samples
        second_term = torch.tensor(0, dtype=torch.float)
        for sample_i in range(n):
            for sample_j in range(n):
                second_term = _ged_dist_func(samples[b, sample_i, ...], samples[b, sample_j, ...])
        second_term = (1 / (n ** 2)) * second_term.float()

        # Average GED between ground truths
        third_term = torch.tensor(0, dtype=torch.float)
        for gt_i in range(m):
            for gt_j in range(m):
                third_term += _ged_dist_func(targets[b, gt_i, ...], targets[b, gt_j, ...])
        third_term = (1 / (m ** 2)) * third_term.float()

        geds[b] = first_term - second_term - third_term

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

    for b in range(batch_size):
        num_pairs = 0
        for i, j in combinations(list(range(n)), 2):
            inp1 = samples[b, i, ...]
            inp2 = samples[b, j, ...]

            # Compute number of pixels intersecting
            intersection = inp1 * inp2
            sum_intersection = intersection.view(num_labels, -1).sum(1)
            # Compute number of pixels in union
            union = (inp1 + inp2)
            sum_union = (union > 0).view(num_labels, -1).sum(1)  # Will exclude double-intersection as well

            for label_i in range(num_labels):
                sum_i = sum_intersection[label_i].float()
                sum_u = sum_union[label_i].float()

                if sum_i == 0 and sum_u == 0:
                    dice_per_label[b, label_i] += torch.tensor(1, dtype=torch.float)

                elif (sum_i > 0 and sum_u == 0) or \
                        (sum_i == 0 and sum_u > 0):
                    dice_per_label[b, label_i] += torch.tensor(0, dtype=torch.float)

                else:
                    dice_per_label[b, label_i] += sum_i.float() / sum_u

            num_pairs += 1

        dice_per_label[b, :] /= num_pairs

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

    for b in range(batch_size):
        # Size of intersection and union is [NUM_CHANNELS x H x W]
        intersection = torch.ones(samples.shape[2:]).long().to(_device)
        union = torch.zeros(samples.shape[2:]).long().to(_device)

        # Intersection and union over all samples
        for i in range(n):
            intersection *= samples[b, i, ...].long()
            union += samples[b, i, ...].long()

        sum_intersection = intersection.view(num_labels, -1).sum(1)
        sum_union = (union > 0).view(num_labels, -1).sum(1)  # Will exclude double-intersection as well

        dice_per_label[b, ...] = (sum_intersection.float() + 1e-6) / (sum_union.float() + 1e-6)

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
    N = samples.shape[0]

    mean_samples = samples.mean(0)

    gamma_maps = torch.zeros((N, samples.shape[2], samples.shape[3]), dtype=torch.float).to(_device)
    for i in range(N):
        gamma_maps[i, ...] += _pixel_wise_xent(samples[i, ...], mean_samples)
    gamma_map = gamma_maps.mean(0)

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
    for b in range(batch_size):
        batch_samples = samples[b, ...].float()
        batch_g_truths = g_truths[b, ...].float()

        gamma_map_ss = pixel_wise_ce_samples(batch_samples)

        E_sy_arr = torch.zeros((M, N, batch_samples.size()[2], batch_samples.size()[3]))
        for j in range(M):
            for i in range(N):
                E_sy_arr[j, i, ...] = _pixel_wise_xent(batch_samples[i, ...], batch_g_truths[j, ...])

        E_sy = E_sy_arr.mean(dim=1)

        nccs = torch.zeros(M).float()
        for j in range(M):
            nccs[j] = _ncc(gamma_map_ss, E_sy[j, ...])

        batch_nccs[b] = nccs.mean()

    return torch.mean(batch_nccs).cpu().item()


def _ged_dist_func(inp1: torch.Tensor, inp2: torch.Tensor):
    inp1 = inp1.float()
    inp2 = inp2.float()

    inp_size = inp1.size()
    num_labels = inp_size[0]

    intersection = inp1 * inp2
    sum_intersection = intersection.view(num_labels, -1).sum(1)
    union = (inp1 + inp2)
    sum_union = (union > 0).view(num_labels, -1).sum(1)  # Will exclude double-intersection as well

    per_label_iou = torch.zeros(num_labels).float()
    for label_i in range(inp_size[0]):
        sum_i = sum_intersection[label_i]
        sum_u = sum_union[label_i]

        if sum_i == 0 and sum_u == 0:
            per_label_iou[label_i] = torch.tensor(1, dtype=torch.float)
        elif (sum_i > 0 and sum_u == 0) or (sum_i == 0 and sum_u > 0):
            per_label_iou[label_i] = torch.tensor(0, dtype=torch.float)
        else:
            per_label_iou[label_i] = sum_i.float() / sum_u

    return 1 - per_label_iou.mean()


def _pixel_wise_xent(sample, gt, eps=1e-8):
    log_sample = torch.log(sample + eps)
    return -1 * torch.sum(gt * log_sample, 0)


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
