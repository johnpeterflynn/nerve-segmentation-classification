import torch
from utils import visualization
import model.metric as metrics
import random


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
    ged = metrics.ged(samples, targets).cpu().numpy()
    ged = "{0:.2f}".format(ged)

    dice_agreement_S = metrics.dice_agreement_in_samples(samples)
    dice_agreement_S = "{0:.2f}".format(dice_agreement_S)

    iou_S = metrics.iou_samples_per_label(samples)
    iou_S = "{0:.2f}".format(iou_S)

    var_ncc_S = metrics.variance_ncc_samples(samples, targets)
    var_ncc_S = "{0:.2f}".format(var_ncc_S)

    return [ged, dice_agreement_S, iou_S, var_ncc_S]


if __name__ == '__main__':
    IMG_SIZE = (1, 128, 128)
    BATCH_SIZE = 5
    NUM_GT = 4
    NUM_S = 3  # number of samples
    binary_images = generate_binary_imgs_random(count=20, img_size=IMG_SIZE)
    random.seed(100)

    result_images = []
    calculated_metrics = []

    # Fixed images
    gts = torch.stack([random.choice(binary_images) for _ in range(NUM_GT)]).unsqueeze(dim=0)
    fixImg = torch.zeros(IMG_SIZE)
    fixImg[0, 40:60, 40:60] = 1
    samples = torch.stack([fixImg, fixImg, fixImg]).unsqueeze(dim=0)
    calculated_metrics.append(calculate_metrics(samples, gts))
    result_images.append(torch.cat([gts, samples], dim=1))

    for i in range(BATCH_SIZE):
        # Unsqueeze to have 1 as batch size
        gts = torch.stack([random.choice(binary_images) for _ in range(NUM_GT)]).unsqueeze(dim=0)
        samples = torch.stack([random.choice(binary_images) for _ in range(NUM_S)]).unsqueeze(dim=0)

        calculated_metrics.append(calculate_metrics(samples, gts))

        # for visualization we need all images concatenated per batch
        result_images.append(torch.cat([gts, samples], dim=1))

    gt_titles = [f'GT_{i}' for i in range(NUM_GT)]
    s_titles = [f'S_{i}' for i in range(NUM_S)]
    metric_titles = ['GED', 'DICE_S', 'IoU_S', 'VNCC_S']
    titles = gt_titles + s_titles + metric_titles

    result_images = torch.cat(result_images)
    img_metric_grid = visualization.make_image_metric_grid(result_images,
                                                           textList=calculated_metrics,
                                                           titles=titles,
                                                           enable_helper_dots=True)
    visualization.visualize_image_grid(img_metric_grid)

    binary_image1 = binary_images[0].unsqueeze(dim=0).unsqueeze(dim=0)
    binary_image2 = binary_images[1].unsqueeze(dim=0).unsqueeze(dim=0)
