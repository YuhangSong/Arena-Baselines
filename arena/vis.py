import imageio
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd

from .utils import *
from .constants import policy_i2id

sns.set()


def vis_result_matrix(result_matrix):
    if len(np.shape(result_matrix)) == 3:
        for policy_i in range(result_matrix.shape[2]):
            policy_id = policy_i2id(policy_i)

            fig = plt.figure()
            ax = plt.gca()

            sns.heatmap(
                pd.DataFrame(result_matrix[:, :, policy_i]),
                annot=True,
                fmt='.2f',
                linewidths=.05,
                cmap="coolwarm",
                ax=ax,
            )

            plt.close()

            img = get_img_from_fig(fig)

            imageio.imwrite('result_matrix-{}.jpg'.format(policy_id), img)
    else:
        raise NotImplementedError


def get_img_from_fig(fig, dpi=180):
    # define a function which returns an image as numpy array from figure
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=180)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def plot_feature(data, label=None, y_range=None, new_fig=True, fig=None):
    # plot a feature of size(x)
    if new_fig:
        fig = plt.figure()
    ax = plt.gca()
    if y_range is not None:
        ax.set_ylim(y_range)
    ax.plot(np.arange(np.shape(data)[0]), data, label=label)
    if label is not None:
        ax.legend()
    if new_fig:
        plt.close()
    return fig
