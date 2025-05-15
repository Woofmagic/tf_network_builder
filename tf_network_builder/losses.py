# losses.py

import tensorflow as tensorflow

_LOSSES_STRING_BINARY_CROSSENTROPY = "binary crossentropy"
_LOSSES_STRING_BINARY_FOCAL_CROSSENTROPY = "binary focal crossentropy"
_LOSSES_STRING_CATEGORICAL_CROSSENTROPY = "categorical crossentropy"
_LOSSES_STRING_CATEGORICAL_FOCAL_CROSSENTROPY = "categorical focal crossentropy"
_LOSSES_STRING_CATEGORICAL_HINGE = "cateogrical hinge"
_LOSSES_STRING_COSINE_SIMILARITY = "cosine similarity"
_LOSSES_STRING_HINGE = "hinge"
_LOSSES_STRING_HUBER = "huber"
_LOSSES_STRING_KL_DIVERGENCE = "kl divergence"
_LOSSES_STRING_LOG_COSH = "log cosh"
_LOSSES_STRING_LOSS = "loss"
_LOSSES_STRING_MEAN_ABSOLUTE_ERROR = "mean absolute error"
_LOSSES_STRING_MEAN_ABSOLULTE_PERCENTAGE_ERROR = "mean absolute percentage error"
_LOSSES_STRING_MEAN_SQUARED_ERROR = "mean squared error"
_LOSSES_STRING_MEAN_SQUARED_LOGARITHMIC_ERROR = "mean squared logarithmic error"
_LOSSES_STRING_POISSON = "poisson"
_LOSSES_STRING_REDUCTION = "reduction"
_LOSSES_STRING_SPARCE_CATEGORICAL_CROSSENTROPY = "sparce categorical crossentropy"
_LOSSES_STRING_SQUARED_HINGE = "squared hinge"

_ARRAY_OF_ACCEPTABLE_LOSS_FUNCTIONS = [
    _LOSSES_STRING_BINARY_CROSSENTROPY,
    _LOSSES_STRING_BINARY_FOCAL_CROSSENTROPY,
    _LOSSES_STRING_CATEGORICAL_CROSSENTROPY,
    _LOSSES_STRING_CATEGORICAL_FOCAL_CROSSENTROPY,
    _LOSSES_STRING_COSINE_SIMILARITY,
    _LOSSES_STRING_CATEGORICAL_HINGE,
    _LOSSES_STRING_HUBER,
    _LOSSES_STRING_KL_DIVERGENCE,
    _LOSSES_STRING_LOG_COSH,
    _LOSSES_STRING_LOSS,
    _LOSSES_STRING_MEAN_ABSOLUTE_ERROR,
    _LOSSES_STRING_MEAN_ABSOLULTE_PERCENTAGE_ERROR,
    _LOSSES_STRING_MEAN_SQUARED_ERROR,
    _LOSSES_STRING_MEAN_SQUARED_LOGARITHMIC_ERROR,
    _LOSSES_STRING_POISSON,
    _LOSSES_STRING_REDUCTION,
    _LOSSES_STRING_SPARCE_CATEGORICAL_CROSSENTROPY,
    _LOSSES_STRING_SQUARED_HINGE
        ]

_DICTIONARY_MAP_USER_INPUT_TO_KERAS_LOSS = {
    _LOSSES_STRING_BINARY_CROSSENTROPY: tensorflow.keras.losses.BinaryCrossentropy(),
    _LOSSES_STRING_BINARY_FOCAL_CROSSENTROPY: tensorflow.keras.losses.BinaryFocalCrossentropy(),
    _LOSSES_STRING_CATEGORICAL_CROSSENTROPY: tensorflow.keras.losses.CategoricalCrossentropy(),
    _LOSSES_STRING_CATEGORICAL_FOCAL_CROSSENTROPY: tensorflow.keras.losses.CategoricalFocalCrossentropy(),
    _LOSSES_STRING_CATEGORICAL_HINGE: tensorflow.keras.losses.CategoricalHinge(),
    _LOSSES_STRING_COSINE_SIMILARITY: tensorflow.keras.losses.CosineSimilarity(),
    _LOSSES_STRING_HINGE: tensorflow.keras.losses.Hinge(),
    _LOSSES_STRING_HUBER: tensorflow.keras.losses.Huber(),
    _LOSSES_STRING_KL_DIVERGENCE: tensorflow.keras.losses.KLDivergence(),
    _LOSSES_STRING_LOG_COSH: tensorflow.keras.losses.LogCosh(),
    _LOSSES_STRING_LOSS: tensorflow.keras.losses.Loss(),
    _LOSSES_STRING_MEAN_ABSOLUTE_ERROR: tensorflow.keras.losses.MeanAbsoluteError(),
    _LOSSES_STRING_MEAN_ABSOLULTE_PERCENTAGE_ERROR: tensorflow.keras.losses.MeanAbsolutePercentageError(),
    _LOSSES_STRING_MEAN_SQUARED_ERROR: tensorflow.keras.losses.MeanSquaredError(),
    _LOSSES_STRING_MEAN_SQUARED_LOGARITHMIC_ERROR: tensorflow.keras.losses.MeanSquaredLogarithmicError(),
    _LOSSES_STRING_POISSON: tensorflow.keras.losses.Poisson(),
    _LOSSES_STRING_REDUCTION: tensorflow.keras.losses.Reduction(),
    _LOSSES_STRING_SPARCE_CATEGORICAL_CROSSENTROPY: tensorflow.keras.losses.SparseCategoricalCrossentropy(),
    _LOSSES_STRING_SQUARED_HINGE: tensorflow.keras.losses.SquaredHinge(),
}