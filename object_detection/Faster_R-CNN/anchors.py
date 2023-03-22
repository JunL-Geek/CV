import numpy as np
import matplotlib.pyplot as plt

def generate_anchor_base(base_size=16, ratios=[0.5, 1, 2], anchor_scales=[8, 16, 32]):
    """

    :param base_size: ratio of original image to feature map
    :param ratios: aspect ratio
    :param anchor_scales:
    :return:
    """
    anchor_base = np.zeros((len(ratios) * len(anchor_scales), 4), dtype=np.float32)
    for i in range(len(ratios)):
        for j in range(len(anchor_scales)):
            w = base_size * anchor_scales[j] * np.sqrt(ratios[i])
            h = base_size * anchor_scales[j] * np.sqrt(1. / ratios[i])

            index = i * len(anchor_scales) + j
            anchor_base[index, 0] = -w / 2
            anchor_base[index, 1] = -h / 2
            anchor_base[index, 2] = w / 2
            anchor_base[index, 3] = h / 2

    return anchor_base

def _enumerate_shifted_anchor(anchor_base, feat_stride, height, width):
    shift_x = np.arange(0, width * feat_stride, feat_stride)
    shift_y = np.arange(0, height * feat_stride, feat_stride)
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shift = np.stack((shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel(),), axis=1)

    num_anchor_per_center = anchor_base.shape[0]
    num_center = shift.shape[0]

    anchor = anchor_base.reshape(1, num_anchor_per_center, 4) + shift.reshape(num_center, 1, 4)
    anchor = anchor.reshape(num_center * num_anchor_per_center, 4).astype(np.float32)
    return anchor

if __name__ == "__main__":
    anchor_base = generate_anchor_base()
    anchor_all = _enumerate_shifted_anchor(anchor_base, feat_stride=16, height=38, width=38)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.ylim(-300, 900)
    plt.xlim(-300, 900)
    shift_x = np.arange(0, 38 * 16, 16)
    shift_y = np.arange(0, 38 * 16, 16)
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    plt.scatter(shift_x, shift_y)
    box_widths = anchor_all[:, 2] - anchor_all[:, 0]
    box_heights = anchor_all[:, 3] - anchor_all[:, 1]

    for i in range(0, 9):
        rect = plt.Rectangle([anchor_all[i, 0], anchor_all[i, 1]], width=box_widths[i], height=box_heights[i], color='r', fill=False)
        ax.add_patch(rect)
    plt.show()
