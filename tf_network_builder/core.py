# core.py

from .prompts import PROMPT_NUMBER_OF_LAYERS, ERROR_INVALID_LAYER_INPUT, PROMPT_NODES_PER_LAYER, ERROR_INVALID_NODES_INPUT

from .validation import validate_configuration, get_positive_integer, get_layerwise_integers

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
        try:

            # (): Pass the dictionary into the validation function:
            validated = validate_configuration(configuration_dict, self.verbose)

        except Exception as error:

            # (): Too general, yes, but not sure what we put here yet:
            raise Exception("> Error occurred during validation...") from error

        # (): Extract the number of hidden layers in the network:
        self.number_of_hidden_layers = validated["number_of_hidden_layers"]

        # (): Extract the list of number of nodes per layer:
        self.list_of_number_of_nodes_per_layer = validated["nodes_per_layer"]

        # (): Extract the number of activation functions per layer:
        self.list_of_activation_functions_for_each_layer = validated["activation_per_layer"]

        # (): Extract the models' loss/objective function as a string:
        self.model_loss_function = validated["loss_function"]

        # (): Extract an integer representing the number of inputs into the network:
        self.number_of_input_variables = validated["input_dimension"]

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