from ray.rllib.models.tf.tf_action_dist import Categorical
from ray.rllib.utils.annotations import override
from ray.rllib.models import Model, ModelCatalog


class DeterministicCategorical(Categorical):
    """Deterministic version of categorical distribution for discrete action spaces.
    """

    @override(Categorical)
    def _build_sample_op(self):
        return tf.squeeze(tf.argmax(self.inputs, 1), axis=1)


class ShareLayerPolicy(Model):
    def _build_layers_v2(self, input_dict, num_outputs, options):
        # Weights shared with CustomModel1
        with tf.variable_scope(
                tf.VariableScope(tf.AUTO_REUSE, "shared"),
                reuse=tf.AUTO_REUSE,
                auxiliary_name_scope=False):
            last_layer = tf.layers.dense(
                input_dict["obs"], 64, activation=tf.nn.relu, name="fc1")
        last_layer = tf.layers.dense(
            last_layer, 64, activation=tf.nn.relu, name="fc2")
        output = tf.layers.dense(
            last_layer, num_outputs, activation=None, name="fc_out")
        return output, last_layer


ModelCatalog.register_custom_model("ShareLayerPolicy", ShareLayerPolicy)
