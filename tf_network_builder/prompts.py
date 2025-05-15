# prompts.py

# (1): The first prompts that are asked prompt the user to specify the
#  | number of *hidden layers* in the network.
PROMPT_NUMBER_OF_LAYERS = "> How many layers do you want in your ANN architecture?"
ERROR_INVALID_LAYER_INPUT = "> Layer number must be a positive, nonzero integer. Try again."

# (2): The second prompts the user must respond to are about the specification of
#  | the number of nodes *per hidden layer* that must be supplied.
PROMPT_NODES_PER_LAYER = "> Choose how many nodes you want for layer {}."
ERROR_INVALID_NODES_INPUT = "> Number of nodes per layer must be an integer type. Try again."

# (3): The third prompt asks the user to provide layer-wise activation functions.
ERROR_ACTIVATION_MISMATCH = "> Provided activations must match number of layers and use only known functions."
ERROR_UNKNOWN_LOSS = "> Provided loss function is not in the known list."