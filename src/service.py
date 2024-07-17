from dependencies import *
from schemas import Classification_request

def generate_unique_filename(request: Classification_request,
                             directory: str) -> str:
  
  filename = request.fileName
  unique_filename = ''
  
  while True:
    #generate unique uuid 
    unique_string = str(uuid.uuid4().hex)[:4]
    unique_filename = unique_string + ' ' + filename
    
    if is_filename_unique(unique_filename, directory):
      break
    
  return unique_filename

def is_filename_unique(filename: str, directory: str) -> bool:
  
  files_in_directory = os.listdir(directory)
  
  if filename in files_in_directory:
    return False
  else:
    return True

async def save_file_from_request(request: Classification_request, filepath: str) -> None:
  try:
    
    base64_data = request.base64Data
    binary_data = base64.b64decode(base64_data)
    
    #save file to mp3
    with open (filepath+'.mp3', 'wb') as file:
      file.write(binary_data)  
      
  except Exception as e:
    print(f'Error: {str(e)}')
    
def convert_mp3_to_wav(filepath: str) -> None:
  audio = AudioSegment.from_mp3(filepath+'.mp3')
  audio.export(filepath+'.wav', format='wav')    
 
def delete_wavs(directory='../mp3/') -> None:
  # List all files in the directory
  files = os.listdir(directory)

  # Iterate through files and delete those with .wav extension
  for file in files:
      if file.endswith(".wav"):
          file_path = os.path.join(directory, file)
          os.remove(file_path)

def generate_mfcc_from_file(filepath: str, n_mfcc=13, n_fft=2048, hop_length=512) -> np.array:

  SAMPLE_RATE = 22050
  mfccs = []

  # load file with librosa
  signal, sr = librosa.load(filepath+'.wav', sr=SAMPLE_RATE)
  duration = math.floor(librosa.get_duration(y=signal, sr=SAMPLE_RATE))

  number_of_segments = int(duration / 3)
  samples_per_track = SAMPLE_RATE * duration
  number_of_samples_per_segment = int(samples_per_track / number_of_segments)
  expected_number_of_mfcc_vectors_per_segment = 130
    
  for segment in range(number_of_segments):
    
      start_sample = number_of_samples_per_segment * segment
      finish_sample = start_sample + number_of_samples_per_segment
    
      # generate mfcc for a segment
      mfcc = librosa.feature.mfcc(y=signal[start_sample:finish_sample],
                                  sr=sr,
                                  n_fft=n_fft,
                                  n_mfcc=n_mfcc,
                                  hop_length=hop_length) 
      mfcc = mfcc.T
      
      #ensure that mfcc count does not exceed expected value
      if len(mfcc) > expected_number_of_mfcc_vectors_per_segment:
        mfcc = np.delete(mfcc, 1, axis=0)
        
      if len(mfcc) == expected_number_of_mfcc_vectors_per_segment:
        mfccs.append(mfcc.tolist())

  return np.array(mfccs)

def predict(model, X) -> tuple[str, dict[str, int], list[str]]:
  
  list_of_genres = ['blues', 'classical',
                    'country', 'disco',
                    'hiphop', 'jazz',
                    'metal', 'pop',
                    'reggae', 'rock']
  genres_sequence = []
  genres_distribution = {
    'blues': 0,
    'classical': 0,
    'country': 0,
    'disco': 0,
    'hiphop': 0,
    'jazz': 0,
    'metal': 0,
    'pop': 0,
    'reggae': 0,
    'rock': 0
  }
  
  for segment in X:
  
    segment = segment[np.newaxis, ...]
    
    prediction = model.predict(segment)
    
    # extract index with max value
    predicted_index = np.argmax(prediction, axis=1)

    # get the predicted genre from the list
    predicted_genre = list_of_genres[int(predicted_index)]
    genres_distribution[predicted_genre] += 1
    
    # save sequence of genres 
    genres_sequence.append(predicted_genre)
    
  # extract main genre from the dictionary
  genre = get_key_with_max_value(genres_distribution)
  
  return genre, genres_distribution, genres_sequence

def get_key_with_max_value(dict: dict) -> str:
  max_key = None
  max_value = float('-inf')

  # Iterate through key-value pairs
  for key, value in dict.items():
    if value > max_value:
        max_key = key
        max_value = value
  
  return max_key

def tag_mp3_file(filepath: str, 
                 request: Classification_request, 
                 genre: str) -> None:
  
  tags = request.tags
  tags['genre'] = genre
  
  file = music_tag.load_file(filepath + '.mp3')
  
  for tag_name, tag_value in tags.items():
    
    # tag has not been assigned
    if tag_value == '': continue
    
    # year does not work as intended
    if tag_name == 'year': continue
    
    file.remove_tag(tag_name)
    file.append_tag(tag_name, tag_value)
    
  # check if user specified year tag
  if tags['year'] != '':
    file.remove_tag('year')  
    
    # append users year 
    file.raw['year'] = tags['year']
  
  file.save()