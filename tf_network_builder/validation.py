# validation.py

def get_positive_integer(prompt_template, error_message, preprovided_value = None, verbose = False):
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