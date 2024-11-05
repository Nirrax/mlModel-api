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
    
def test_delete_wavs(tmp_path):
    directory = tmp_path / "sub"
    directory.mkdir()
    
    assert len(list(directory.iterdir())) == 0
    
    delete_wavs(directory=directory)
    
    assert len(list(directory.iterdir())) == 0
    
    #generate files
    files = []
    files.append(directory / "file1.wav")
    files.append(directory / "file2.WaV")
    files.append(directory / "file3.png")
    files.append(directory / "file4.mp3")
    files.append(directory / "file5.txt")
    files.append(directory / "file6.wav")
    for file in files:
        file.touch() 
        
    #count files before delete
    assert len(list(directory.iterdir())) == 6
    
    delete_wavs(directory=directory)
    
    #count files after delete
    assert len(list(directory.iterdir())) == 4