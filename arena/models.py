import logging

from ray.rllib.models.tf.tf_action_dist import Categorical
from ray.rllib.utils.annotations import override
from ray.rllib.models import Model, ModelCatalog
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.utils import try_import_tf

tf = try_import_tf()

logger = logging.getLogger(__name__)


class DeterministicCategorical(Categorical):
    """Deterministic version of categorical distribution for discrete action spaces.
    """

    @override(Categorical)
    def _build_sample_op(self):
        return tf.squeeze(tf.argmax(self.inputs, 1), axis=1)


class ArenaPolicy(TFModelV2):
    """Multi-agent policy that supports:
    1, weights sharing between policies;
    2, centralized critic;
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        super(CentralizedCriticModel, self).__init__(
            obs_space, action_space, num_outputs, model_config, name)

        self.action_model = FullyConnectedNetwork(
            Box(low=0, high=1, shape=(6, )),  # one-hot encoded Discrete(6)
            action_space,
            num_outputs,
            model_config,
            name + "_action")
        self.register_variables(self.action_model.variables())

        self.value_model = FullyConnectedNetwork(obs_space, action_space, 1,
                                                 model_config, name + "_vf")
        self.register_variables(self.value_model.variables())

    def forward(self, input_dict, state, seq_lens):
        self._value_out, _ = self.value_model({
            "obs": input_dict["obs_flat"]
        }, state, seq_lens)
        return self.action_model({
            "obs": input_dict["obs"]["own_obs"]
        }, state, seq_lens)

    def value_function(self):
        return tf.reshape(self._value_out, [-1])

    def _build_layers_v2(self, input_dict, num_outputs, options):

        policies_shared_scope = "shared_by_{}".format(
            str(
                self.options["custom_options"]
                ["shared_scope"]
            ).replace(", ", ",")
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
            # layers here are shared across policies in policies_shared_scope
            last_layer = tf.layers.dense(
                input_dict["obs"], 64, activation=tf.nn.relu, name="fc1")

        # layers here are owned by each individual policy
        last_layer = tf.layers.dense(
            last_layer, 64, activation=tf.nn.relu, name="fc2")
        output = tf.layers.dense(
            last_layer, num_outputs, activation=None, name="fc_out")

        # for defining more models, refer to https://github.com/ray-project/ray/tree/master/rllib/models/tf

        return output, last_layer


ModelCatalog.register_custom_model("ArenaPolicy", ArenaPolicy)
