import pytest, asyncio
from dependencies import *
from service import *
from schemas import Classification_request

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
    
def test_tag_mp3_file(tmp_path):
    directory = tmp_path / "sub"
    directory.mkdir()
    
    assert len(list(directory.iterdir())) == 0
    
    #create an empty mp3 file
    empty_mp3 = AudioSegment.silent(duration=250)
    empty_mp3.export((directory / "file.mp3"), format="mp3")
    
    file_mp3 = music_tag.load_file(directory / "file.mp3")
    
    #test if tags are nonexisting
    assert str(file_mp3['artist']) == ''
    assert str(file_mp3['title']) == ''
    assert str(file_mp3['album']) == ''
    assert str(file_mp3['year']) == ''
    assert str(file_mp3['genre']) == ''
    file_mp3.save()
    
    tags = {
        'artist': 'andrzej',
        'title': 'kowalski',
        'album': 'Miodowe lata',
        'year': 2005
    }
    
    tag_mp3_file(str(directory / "file"), tags, 'rock')
    
    #test if tags are matching
    file_mp3 = music_tag.load_file(directory / "file.mp3")
    assert str(file_mp3['artist']) == 'andrzej'
    assert str(file_mp3['title']) == 'kowalski'
    assert str(file_mp3['album']) == 'Miodowe lata'
    assert str(file_mp3['year']) == '2005' 
    
@pytest.mark.asyncio    
async def test_save_file_from_request(tmp_path):
    directory = tmp_path / 'sub'
    directory.mkdir()
    
    assert len(list(directory.iterdir())) == 0
    
    empty_mp3 = AudioSegment.silent(duration=250)
    empty_mp3.export((directory / "file.mp3"), format="mp3")
    
    with open(directory / "file.mp3", "rb") as mp3_file:
        encoded_mp3 = base64.b64encode(mp3_file.read())
    
    request = Classification_request(fileName="", tags={}, base64Data=encoded_mp3)
    await save_file_from_request(request, str((directory / "newFile")))
    
    assert os.path.getsize(directory / "file.mp3") == os.path.getsize(directory / "newFile.mp3")
    