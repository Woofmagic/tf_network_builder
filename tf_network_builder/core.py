# core.py

from .prompts import PROMPT_NUMBER_OF_LAYERS, ERROR_INVALID_LAYER_INPUT, PROMPT_NODES_PER_LAYER, ERROR_INVALID_NODES_INPUT

from .validation import get_positive_integer, get_layerwise_integers

class TFNetworkBuilder:
    """
    TensorFlow Network Builder
    """
    def __init__(self, configuration = None, verbose = True):
        self.configuration_mode = configuration is not None

        self.number_of_hidden_layers = None
        self.list_of_number_of_nodes_per_layer = []
        self.list_of_activation_functions_for_each_layer = []
        self.model_loss_function = None
        self.number_of_input_variables = 0

        self.verbose = verbose

        if configuration:
            self._initialize_from_config(configuration)

    def _initialize_from_config(self, configuration_dict: dict):
        validated = validate_config(configuration_dict)

        self.number_of_hidden_layers = validated["num_layers"]
        self.list_of_number_of_nodes_per_layer = validated["nodes_per_layer"]
        self.list_of_activation_functions_for_each_layer = validated["activations"]
        self.model_loss_function = validated["loss"]
        self.number_of_input_variables = validated["input_dim"]

        if self.verbose:
            print(f"> Network initialized from config: {validated}")


    def obtain_number_of_ann_hidden_layers(self):
        self.number_of_hidden_layers = get_positive_integer(
        prompt_template = PROMPT_NUMBER_OF_LAYERS,
        error_message = ERROR_INVALID_LAYER_INPUT,
        verbose = self.verbose)

    def obtain_nodes_per_layer(self):
        if self.number_of_hidden_layers is None:
            raise ValueError("Number of hidden layers must be set before getting nodes per layer.")

        self.list_of_number_of_nodes_per_layer = get_layerwise_integers(
            prompt_template = PROMPT_NODES_PER_LAYER,
            error_message = ERROR_INVALID_NODES_INPUT,
            number_of_hidden_layers = self.number_of_hidden_layers,
            verbose = self.verbose
        )