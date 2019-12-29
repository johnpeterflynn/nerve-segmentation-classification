import torch
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import matplotlib as mpl
import matplotlib.cm as cm
import model.metric as metrics



def make_image_metric_grid(imgTensor: torch.Tensor,
                           textList: list = None,
                           titles: list = None,
                           margin=(10, 10),
                           cmap='gray',
                           bgColor=(255, 153, 51),
                           heatMaps: torch.Tensor = None,
                           enable_helper_dots=False):
    """

    :param imgTensor: A tensor object with one of the following shapes:
            [ROW_COUNT, COLUMN_COUNT, NUM_CHANNELS, H, W]   # multiple rows with multiple images
            or
            [COLUMN_COUNT, NUM_CHANNELS, H, W]  # single row
            or
            [NUM_CHANNELS, H, W]    # single image
    :param textList: Text Data to show after images, shape should be 2D list with no gaps
    :param titles: Column titles, list of shape [COLUMN_COUNT]
    :param margin: Margin between images in pixels [VERTICAL_IN_PX, HORIZONTAL_IN_PX]
    :param cmap: Same as Matplotlib CMAP
    :param bgColor: Background color in RGB format
    :param heatMaps: A tensor object with the following shape:
            [ROW_COUNT, H, W]   # multiple rows with one heatmap
    """
    N_rows, N_cols, _, H, W = 0, 0, 0, 0, 0
    N_img_cols, N_text_cols, N_heatmap_cols = 0, 0, 0
    titles_enabled = titles is not None
    H_TITLES = 40 if titles_enabled else 0
    heatMaps_enabled = heatMaps is not None

    if len(imgTensor.shape) == 5:
        N_rows, N_img_cols, N_channels, H, W = imgTensor.shape
    elif len(imgTensor.shape) == 4:
        N_img_cols, N_channels, H, W = imgTensor.shape
        N_rows = 1
        imgTensor = imgTensor.unsqueeze(dim=0)
    elif len(imgTensor.shape) == 3:
        N_channels, H, W = imgTensor.shape
        N_rows, N_img_cols = 1, 1
        imgTensor = imgTensor.unsqueeze(dim=0).unsqueeze(dim=0)

    if textList is not None:
        N_text_cols = len(textList[0])

    if heatMaps_enabled:
        N_heatmap_cols = 1

    N_cols = N_img_cols + N_text_cols + N_heatmap_cols

    final_H = N_rows * H + (N_rows - 1) * margin[0] + H_TITLES
    final_W = N_cols * W + (N_cols - 1) * margin[1]

    v_data = torch.zeros((3, final_H, final_W))
    v_data[0, :, ...] = bgColor[2] / 255
    v_data[1, :, ...] = bgColor[1] / 255
    v_data[2, :, ...] = bgColor[0] / 255

    textFont = ImageFont.truetype("resources/arial.ttf", 24)

    if titles_enabled:
        v_data[:, 0:H_TITLES, :] = torch.ones((3, H_TITLES, final_W))

        for u_idx in range(N_cols):
            if u_idx >= len(titles):
                continue

            u_start = u_idx * (W + margin[1])
            u_end = u_start + W

            v_data[:, 0:H_TITLES, u_start:u_end] = _get_textImg(titles[u_idx], H_TITLES, W, textFont)

    for v_idx in range(N_rows):
        for u_idx in range(N_cols):
            v_start = v_idx * (H + margin[0]) + H_TITLES
            v_end = v_start + H
            u_start = u_idx * (W + margin[1])
            u_end = u_start + W

            if u_idx < N_img_cols:  # image
                v_data[:, v_start:v_end, u_start:u_end] = imgTensor[v_idx, u_idx]

                if enable_helper_dots:
                    v_data[0, v_start:v_end:32, u_start:u_end:2] = 1
                    v_data[0, v_start:v_end:2, u_start:u_end:32] = 1
                    v_data[1, v_start:v_end:32, u_start:u_end:2] = 0
                    v_data[1, v_start:v_end:2, u_start:u_end:32] = 0
                    v_data[2, v_start:v_end:32, u_start:u_end:2] = 0
                    v_data[2, v_start:v_end:2, u_start:u_end:32] = 0

            elif u_idx < N_img_cols + N_heatmap_cols:
                heatMap = heatMaps[v_idx]
                heatMap = heatMap.cpu().numpy()

                norm = mpl.colors.Normalize(vmin=0, vmax=heatMap.max())
                cmap = cm.viridis
                m = cm.ScalarMappable(norm=norm, cmap=cmap)

                rgb_ce_map = m.to_rgba(heatMap)[:, :, :3]
                rgb_ce_map = torch.from_numpy(rgb_ce_map).permute(2, 0, 1)
                v_data[:, v_start:v_end, u_start:u_end] = rgb_ce_map

            else:   # text
                text = textList[v_idx][u_idx - N_img_cols]
                text = str(text)
                v_data[:, v_start:v_end, u_start:u_end] = _get_textImg(text, H, W, textFont)

    return v_data


def samples_heatmap(samples, algo=2):
    """
            Builds Samples Heatmap which pictures variance over samples

        :param samples: Tensor of shape [BATCH_SIZE x NUM_SAMPLES x NUM_CHANNELS x H x W]
            where C is number of classes
        :param algo: Integer which selectes algorithm for heatmaps,
            algo = 1    =>  Computes variance among samples for each pixel
            algo = 2    =>  Generates Cross-Entropy Gamma maps from PhiSegNet paper
        :return: Tensor of shape [BATCH_SIZE x H x W]
        """

    if algo == 1:
        variance = torch.var(samples, dim=1, keepdim=True)
        variance.squeeze_()
        return variance
    elif algo == 2:
        ce_map = metrics.pixel_wise_ce_samples(samples)
        return ce_map

def visualize_image_grid(image_grid):
    if image_grid.shape[0] == 3:
        image_grid = image_grid.permute(1, 2, 0)
    image_grid = image_grid.cpu().numpy()

    plt.imshow(image_grid)
    plt.axis('off')
    plt.show()


def _get_textImg(text, H, W, font):
    txtImg = Image.new('RGB', (W, H), color=(255, 255, 255))
    textDraw = ImageDraw.Draw(txtImg)
    w, h = textDraw.textsize(text, font=font)
    textDraw.text(((W - w) / 2, (H - h) / 2), text, fill=(0, 0, 0), font=font)

    txtImg = np.array(txtImg) / 255
    return torch.from_numpy(txtImg).permute(2, 0, 1)
