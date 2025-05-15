# setup.py
from setuptools import setup, find_packages

# (): Use setup() to define various library parameters:
setup(

    # (): The name of the library: 
    name = "tf_network_builder",

    # (): The current version of the library:
    version = "0.1.0",

    # (): Execute find_packages() to list the required packages...
    packages = find_packages(),

    # (): Specify a list of dependencies for the library:
    install_requires = [
        "tensorflow"
        ],

    # (): Author name:
    author = "Woofmagic",

    # (): Library description:
    description = "Design a 'customized' feedforward DNN using TensorFlow",
)
