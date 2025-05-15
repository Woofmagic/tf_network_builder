# core.py

from .prompts import PROMPT_NUMBER_OF_LAYERS, ERROR_INVALID_LAYER_INPUT

from .validation import get_positive_integer

class TFNetworkBuilder:
    """
    TensorFlow Network Builder
    """
    def __init__(self, verbose = True):
        self.verbose = verbose
        self.number_of_hidden_layers = None

    def obtain_number_of_ann_hidden_layers(self):
        self.number_of_hidden_layers = get_positive_integer(
        prompt = PROMPT_NUMBER_OF_LAYERS,
        error_message = ERROR_INVALID_LAYER_INPUT,
        verbose = self.verbose)
