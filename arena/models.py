import logging

from ray.rllib.models.tf.tf_action_dist import Categorical
from ray.rllib.utils.annotations import override
from ray.rllib.models import Model, ModelCatalog
from ray.rllib.utils import try_import_tf

tf = try_import_tf()

logger = logging.getLogger(__name__)


class DeterministicCategorical(Categorical):
    """Deterministic version of categorical distribution for discrete action spaces.
    """

    @override(Categorical)
    def _build_sample_op(self):
        return tf.squeeze(tf.argmax(self.inputs, 1), axis=1)


class ShareLayerPolicy(Model):
    def _build_layers_v2(self, input_dict, num_outputs, options):
        policies_shared_scope = "shared_by_{}".format(
            str(self.options["custom_options"]
                ["shared_scope"]).replace(", ", ",")
        )
        logger.info("policies_shared_scope={}".format(
            policies_shared_scope
        ))
        with tf.variable_scope(
            tf.VariableScope(
                tf.AUTO_REUSE,
                policies_shared_scope
            ),
            reuse=tf.AUTO_REUSE,
            auxiliary_name_scope=False
        ):
            # these layer are in the policies_shared_scope
            last_layer = tf.layers.dense(
                input_dict["obs"], 64, activation=tf.nn.relu, name="fc1")

        # these layers are owned by each individual policy
        last_layer = tf.layers.dense(
            last_layer, 64, activation=tf.nn.relu, name="fc2")
        output = tf.layers.dense(
            last_layer, num_outputs, activation=None, name="fc_out")

        return output, last_layer


ModelCatalog.register_custom_model("ShareLayerPolicy", ShareLayerPolicy)
