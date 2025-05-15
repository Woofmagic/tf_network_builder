# examples/config_driven.py

from tf_network_builder.core import TFNetworkBuilder


# Example (1):
# | We enlist a dictionary of configuration parameters that correctly
# | map to settings in the network builder class.
try:
    example_1_config_dictionary = {
        "number_of_hidden_layers": 3,
        "nodes_per_layer": [64, 32, 16],
        "activation_per_layer": ["relu", "relu", "sigmoid"],
        "loss_function": "binary crossentropy",
        "input_dimension": 10
    }

    example_1_tf_network = TFNetworkBuilder(
        configuration = example_1_config_dictionary,
        verbose = True)
    
except:
    print("> This error will NOT be shown --- everything is correct above.")

# Example (2):
# | We enlist a dictionary with a bad number of hidden layers:
try:
    example_2_config_dictionary = {
        "number_of_hidden_layers": -3,
        "nodes_per_layer": [64, 32, 16],
        "activation_per_layer": ["relu", "relu", "sigmoid"],
        "loss_function": "binary crossentropy",
        "input_dimension": 4
    }

    example_2_tf_network = TFNetworkBuilder(
        configuration = example_2_config_dictionary,
        verbose = True)

except Exception as error:
    print(f"> This error will be a ValueError --- the number of hidden layers must be positive definite:\n> {error}")

# Example (3):
# | We provide a mismatch between the number of hidden layers and the 
# | length of the array `nodes_per_layer`. These must obviously match.
try:
    example_3_config_dictionary = {
        "number_of_hidden_layers": 5,
        "nodes_per_layer": [2, 2, 2, 2, 2, 2, 2, 2],
        "activation_per_layer": ["relu", "relu", "relu", "relu", "relu"],
        "loss_function": "binary crossentropy",
        "input_dimension": 2
    }

    example_3_tf_network = TFNetworkBuilder(
        configuration = example_3_config_dictionary,
        verbose = True)

except Exception as error:
    print(f"> This error will be a ValueError --- the number of hidden layers must match the length of the nodes_per_layer array:\n> {error}")

# model = builder.build_model()
# model.summary()
