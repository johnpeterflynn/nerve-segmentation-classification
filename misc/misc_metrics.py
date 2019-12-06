import torch
from utils import visualization
import model.metric as metrics
import random
import time


def generate_binary_imgs(img_size=(1, 128, 128)):
    images = []

    images.append(torch.zeros(img_size))    # Empty image

    img = torch.zeros(img_size)
    img[0, 32:96, 32:96] = 1
    images.append(img)

    img = torch.zeros(img_size)
    img[0, 64:, 64:] = 1
    images.append(img)

    img = torch.zeros(img_size)
    img[0, 40:50, 40:50] = 1
    images.append(img)

    img = torch.zeros(img_size)
    img[0, 40:50, 60:70] = 1
    images.append(img)

    img = torch.zeros(img_size)
    img[0, 40:50, 60:70] = 1
    images.append(img)

    return torch.stack(images)


def generate_binary_imgs_random(count=10, img_size=(1, 128, 128)):
    images = []
    MIN_H = 30
    MIN_W = 30

    images.append(torch.zeros(img_size))  # Empty image

    for i in range(count):
        v_start = random.randrange(img_size[1] - MIN_H)
        u_start = random.randrange(img_size[2] - MIN_W)
        v_end = random.randrange(v_start + MIN_H, img_size[1])
        u_end = random.randrange(u_start + MIN_W, img_size[2])

        try:
            assert 0 <= v_start < v_end < img_size[1]
            assert 0 <= u_start < u_end < img_size[2]
        except AssertionError:
            print(v_start, v_end, u_start, u_end)
            raise Exception('Oops!')

        img = torch.zeros(img_size)
        img[0, v_start:v_end, u_start:u_end] = 1
        images.append(img)

    return torch.stack(images)


def calculate_metrics(samples, targets):
    start_t = time.time()
    ged = metrics.ged(samples, targets)
    ged = "{0:.2f}".format(ged)

    dice_agreement_S = metrics.dice_agreement_in_samples(samples)
    dice_agreement_S = "{0:.2f}".format(dice_agreement_S)

    iou_S = metrics.iou_samples_per_label(samples)
    iou_S = "{0:.2f}".format(iou_S)

    var_ncc_S = metrics.variance_ncc_samples(samples, targets)
    var_ncc_S = "{0:.2f}".format(var_ncc_S)

    end_t = time.time()
    print("Execution time: ", (end_t - start_t))

    return [ged, dice_agreement_S, iou_S, var_ncc_S]

def test_batched(batch_size, num_gt, num_s, binary_images):
    print("Running in batched mode")
    gts = []
    samples = []
    for i in range(batch_size):
        # Unsqueeze to have 1 as batch size
        gts_i = torch.stack([random.choice(binary_images) for _ in range(num_gt)]).unsqueeze(dim=0)
        samples_i = torch.stack([random.choice(binary_images) for _ in range(num_s)]).unsqueeze(dim=0)

        gts.append(gts_i)
        samples.append(samples_i)

        # for visualization we need all images concatenated per batch
        # result_images.append(torch.cat([gts, samples], dim=1))

    gts = torch.cat(gts)
    samples = torch.cat(samples)

    print(calculate_metrics(samples, gts))


def test_single(count, img_size, num_gt, num_s, binary_images):
    print("Running in single mode")
    result_images = []
    calculated_metrics = []

    # Fixed images
    gts = torch.stack([random.choice(binary_images) for _ in range(num_gt)]).unsqueeze(dim=0)
    fixImg = torch.zeros(img_size)
    fixImg[0, 40:60, 40:60] = 1
    samples = torch.stack([fixImg, fixImg, fixImg]).unsqueeze(dim=0)
    calculated_metrics.append(calculate_metrics(samples, gts))
    result_images.append(torch.cat([gts, samples], dim=1))

    gts = []
    samples = []
    for i in range(count):
        # Unsqueeze to have 1 as batch size
        gts_i = torch.stack([random.choice(binary_images) for _ in range(num_gt)]).unsqueeze(dim=0)
        samples_i = torch.stack([random.choice(binary_images) for _ in range(num_s)]).unsqueeze(dim=0)

        gts.append(gts_i)
        samples.append(samples_i)

        calculated_metrics.append(calculate_metrics(samples_i, gts_i))

        # for visualization we need all images concatenated per batch
        result_images.append(torch.cat([gts_i, samples_i], dim=1))

    gt_titles = [f'GT_{i}' for i in range(num_gt)]
    s_titles = [f'S_{i}' for i in range(num_s)]
    metric_titles = ['GED', 'DICE_S', 'IoU_S', 'VNCC_S']
    titles = gt_titles + s_titles + metric_titles

    result_images = torch.cat(result_images)

    img_metric_grid = visualization.make_image_metric_grid(result_images,
                                                           textList=calculated_metrics,
                                                           titles=titles,
                                                           enable_helper_dots=True)
    visualization.visualize_image_grid(img_metric_grid)


if __name__ == '__main__':
    IMG_SIZE = (1, 128, 128)
    BATCH_SIZE = 5
    NUM_GT = 4
    NUM_S = 3  # number of samples
    binary_images = generate_binary_imgs_random(count=20, img_size=IMG_SIZE)
    random.seed(100)

    test_batched(BATCH_SIZE, NUM_GT, NUM_S, binary_images)
    test_single(BATCH_SIZE, IMG_SIZE, NUM_GT, NUM_S, binary_images)
