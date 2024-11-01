import pytest
from service import *

@pytest.mark.parametrize(
    "input_dict, expected",
    [
        ({"a": 1, "b": 5, "c": 3, "d": 2}, "b"),
        ({"a": 10, "b": 5, "c": 30, "d": 20}, "c"),
        ({"a": 1, "b": 1, "c": 1, "d": 1}, "a"),
        ({}, None),                                  
        ({"a": 1}, "a"),                
        ({"a": -1, "b": -5, "c": -3}, "a")
    ]
)
def test_get_key_with_max_value(input_dict, expected):
    assert get_key_with_max_value(input_dict) == expected

def test_is_filename_unique():
    assert is_filename_unique("file1", []) is True
    assert is_filename_unique("file1", ["file1"]) is False
    assert is_filename_unique("file1", ["file1","file2"]) is False
    assert is_filename_unique("file1", ["file2"]) is True
    assert is_filename_unique("", []) is False