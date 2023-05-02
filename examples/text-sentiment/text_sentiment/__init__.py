import os

from . import data_model, runtime_model
import caikit

# Give the path to the `config.yml`
CONFIG_PATH = os.path.realpath(
    os.path.join(os.path.dirname(__file__), "config.yml")
)

caikit.configure(CONFIG_PATH)

