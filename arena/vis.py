import imageio
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd

from .utils import *
from .constants import policy_i2id

sns.set()


def vis_result_matrix(result_matrix, log_path):

    if len(np.shape(result_matrix)) == 3:

        # where there are two agents,
        # the 0 dimension is loading different checkpoints to agent_0
        # the 1 dimension is loading different checkpoints to agent_1
        # the 3 dimension is episode reward of different agents

        for policy_i in range(result_matrix.shape[2]):

            policy_id = policy_i2id(policy_i)

            fig = plt.figure()
            ax = plt.axes()
            sns.heatmap(
                pd.DataFrame(result_matrix[:, :, policy_i]),
            )
            ax.set_title('result_matrix')
            ax.set_xlabel('policy_0 checkpoints')
            ax.set_ylabel('policy_1 checkpoints')
            plt.close()

            img = get_img_from_fig(fig)

            save_img(
                img=img,
                dir='{}/result_matrix-{}_perspective.jpg'.format(
                    log_path,
                    policy_id,
                ),
            )
    else:

        # TODO: visulize result_matrix generated from other settings
        # For example, a 3T1P game would generate a result_matrix with 4 dimensions.
        # How to visualize it?
        raise NotImplementedError


def get_img_from_fig(fig, dpi=180):
    """Returns an image as numpy array from figure
    """
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=180)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def save_img(img, dir):
    imageio.imwrite(dir, img)
