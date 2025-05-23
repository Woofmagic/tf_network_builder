# validation.py


from .prompts import ERROR_INVALID_LAYER_INPUT, ERROR_INVALID_NODES_INPUT, ERROR_ACTIVATION_MISMATCH, ERROR_UNKNOWN_LOSS
from .losses import _DICTIONARY_MAP_USER_INPUT_TO_KERAS_LOSS
from .activations import _LIST_OF_ACCEPTABLE_ACTIVATION_FUNCTIONS

def validate_configuration(configuration_dictionary: dict, verbose: bool):
    """
    ## Description:
    Validate user's dict of initialization parameters.
    
    ## Parameters:
    configuration_dictionary (dict):
        a dict of user-requests DNN parameters that 
        will be used to attempt to build a TF network.

    verbose (boolean):
        Do you want to see all output of this function evaluation?

    ## Notes
    
    """
    
    try:
        number_of_hidden_layers = int(configuration_dictionary["number_of_hidden_layers"])
        if number_of_hidden_layers <= 0:
            raise ValueError(ERROR_INVALID_LAYER_INPUT)
        
    except Exception as error:
        raise ValueError(ERROR_INVALID_LAYER_INPUT) from error

    # (): We now retrieve the *list* of integers characterising nodes per hidden layer:
    nodes_per_layer = configuration_dictionary.get("nodes_per_layer")

    # (): We now actually valid the *list* of *integers* that is provided in the dictionary:
    if not (isinstance(nodes_per_layer, list) and len(nodes_per_layer) == number_of_hidden_layers and all(isinstance(node_number, int) for node_number in nodes_per_layer)):
        raise ValueError(ERROR_INVALID_NODES_INPUT)
    
    else:
        if verbose:
            print("> Validated list of nodes per layer!")

    proposed_activation_functions = configuration_dictionary.get("activation_per_layer")

    if not (isinstance(proposed_activation_functions, list) and len(proposed_activation_functions) == number_of_hidden_layers and all(activation_function in _LIST_OF_ACCEPTABLE_ACTIVATION_FUNCTIONS for activation_function in proposed_activation_functions)):
        raise ValueError(ERROR_ACTIVATION_MISMATCH)
    
    else:
        if verbose:
            print("> Validated list of activations!")

    proposed_loss_function = configuration_dictionary.get("loss_function")

    if proposed_loss_function not in _DICTIONARY_MAP_USER_INPUT_TO_KERAS_LOSS:
        raise ValueError(ERROR_UNKNOWN_LOSS)
    else:
        if verbose:
            print("> Validated global loss function choice!")

    network_input_dimension = configuration_dictionary.get("input_dimension")

    if not isinstance(network_input_dimension, int) or network_input_dimension <= 0:
        raise ValueError("> Input dimension must be a positive integer.")

    return {
        "number_of_hidden_layers": number_of_hidden_layers,
        "nodes_per_layer": nodes_per_layer,
        "activation_per_layer": proposed_activation_functions,
        "loss_function": _DICTIONARY_MAP_USER_INPUT_TO_KERAS_LOSS[proposed_loss_function],
        "input_dimension": network_input_dimension,
    }

def get_positive_integer(prompt_template, error_message, preprovided_value = None, verbose: bool = False):
    """
    ## Description:
    Obtain the number of hidden layers for the ANN from the user.
    
    ## Parameters:
    prompt_template (string):
        queue the CLI prompt

    error_message (string):
        queue the corresponding error message

    preprovided_value (string):
        if the use wants to specify a dictionary of 
        customized hyperparameters specifying a given DNN
        architecture, then we allow this string argument
        to bypass CLI interactivity
    
    verbose (boolean):
            Do you want to see all output of this function evaluation?
            
    ## Notes
    
    (1) We just need to obtain a nonzero, positive integer that
        represents the number of layers in the ANN.

    (2) https://stackoverflow.com/a/23294659 -> For a healthy way
        to construct a while loop like this.
    """
    if preprovided_value is not None:

        if isinstance(preprovided_value, int) and preprovided_value > 0:
            if verbose:
                print(f"> Using pre-provided value: {preprovided_value}")
            return preprovided_value
        
        else:
            raise ValueError(error_message)

    while True:

        try:

            number_of_ann_layers = int(input(prompt_template))
            if number_of_ann_layers <= 0:
                print(error_message)
                continue

            if verbose:
                print(f"> Received input: {number_of_ann_layers} ({type(number_of_ann_layers)}).")
            return number_of_ann_layers
        
        except ValueError:
            print(error_message)

def get_layerwise_integers(prompt_template, error_message, number_of_hidden_layers, preprovided_values = None, verbose = False):
    """
    ## Description:
    Obtain the number of nodes per layer in the ANN.

    ## Parameters:
    prompt_template (string):
        queue the CLI prompt

    error_message (string):
        queue the corresponding error message

    number_of_hidden_layers (int):
        the number of hidden layers in the DNN
        as specified previously

    preprovided_value (string):
        if the use wants to specify a dictionary of 
        customized hyperparameters specifying a given DNN
        architecture, then we allow this string argument
        to bypass CLI interactivity
    
    verbose (boolean):
            Do you want to see all output of this function evaluation?
    
    ## Notes:

    (1) For all each layer, we need to populate it with a number of neurons.
        So, this function is about obtaining a list of intergers that correspond
        to the number of neurons per layer.

    (2) The output is a list of integers.
    """
    if preprovided_values is not None:

        if isinstance(preprovided_values, list):
            if len(preprovided_values) != 0:
                if verbose:
                    print(f"> Using pre-provided values of: {preprovided_values}")
                return preprovided_values
        
        else:
            raise ValueError(error_message)
        
    list_of_number_of_nodes_per_layer = []

    for hidden_layer_index in range(number_of_hidden_layers):
        while True:
            try:
                value = int(input(prompt_template.format(hidden_layer_index + 1)))
                if verbose:
                    print(f"> Received input: {value} ({type(value)}).")
                list_of_number_of_nodes_per_layer.append(value)
                break

            except ValueError:
                print(error_message)

    return list_of_number_of_nodes_per_layer