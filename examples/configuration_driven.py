# examples/config_driven.py

from tf_network_builder.core import TFNetworkBuilder

example_config_dictionary = {
    "number_of_hidden_layers": 3,
    "nodes_per_layer": [64, 32, 16],
    "activation_per_layer": ["relu", "relu", "sigmoid"],
    "loss_function": "binary crossentropy",
    "input_dimension": 10
}

builder = TFNetworkBuilder(
    configuration = example_config_dictionary,
    verbose = True)

# model = builder.build_model()
# model.summary()
