# pattern
from .domain import Model, match_probability, expected_match_count
from .application import train
from .adapters import FileWordSource, JsonModelStore
from .ports import WordSource, ModelStore

