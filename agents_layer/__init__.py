from ray.rllib.models.tf.tf_action_dist import Categorical
from ray.rllib.utils.annotations import override


class DeterministicCategorical(Categorical):
    """Deterministic version of categorical distribution for discrete action spaces."""

    @override(Categorical)
    def _build_sample_op(self):
        return tf.squeeze(tf.argmax(self.inputs, 1), axis=1)
