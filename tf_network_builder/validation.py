# validation.py

def get_positive_integer(prompt, error_message, preprovided_value = None, verbose = False):
    """
    ## Description:
    Obtain the number of hidden layers for the ANN from the user.
    
    ## Parameters:
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

            number_of_ann_layers = int(input(prompt))
            if number_of_ann_layers <= 0:
                print(error_message)
                continue

            if verbose:
                print(f"> Received input: {number_of_ann_layers} ({type(number_of_ann_layers)}).")
            return number_of_ann_layers
        
        except ValueError:
            print(error_message)
