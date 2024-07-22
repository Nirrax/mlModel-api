import pytest
from service import *

@pytest.mark.parametrize(
    "input_dict, expected",
    [
        ({"a": 1, "b": 5, "c": 3, "d": 2}, "b"),          # Regular case
        ({"a": 10, "b": 5, "c": 30, "d": 20}, "c"),       # Another regular case
        ({"a": 1, "b": 1, "c": 1, "d": 1}, "a"),          # Dictionary with equal values
        ({}, None),                                       # Empty dictionary
        ({"a": 1}, "a"),                                  # Single element
        ({"a": -1, "b": -5, "c": -3}, "a")                # Negative values
    ]
)
def test_get_key_with_max_value(input_dict, expected):
    assert get_key_with_max_value(input_dict) == expected
    