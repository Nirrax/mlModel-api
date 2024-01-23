from dependencies import *

def get_key_with_max_value(dict):
  max_key = None
  max_value = float('-inf')

  # Iterate through key-value pairs
  for key, value in dict.items():
    if value > max_value:
        max_key = key
        max_value = value
  
  return max_key

async def save_file_from_request(request, filepath):
  try:
    
    #parse data from request
    data = await request.json()
    base64_data = data.get('file')
    binary_data = base64.b64decode(base64_data)
    
    #save file to mp3
    with open (filepath+'.mp3', 'wb') as file:
      file.write(binary_data)  
      
  except Exception as e:
    print(f'Error: {str(e)}')
    
def convert_mp3_to_wav(filepath, filename):
  
  audio = AudioSegment.from_mp3(filepath+'.mp3')
  audio.export(filename+'.wav', format='wav')    
  
def generate_mfcc_from_file(file_path, n_mfcc=13, n_fft=2048, hop_length=512):

  SAMPLE_RATE = 22050
  mfccs = []

  # load file with librosa
  signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)
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

def predict(model, X):
  
  list_of_genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
  
  genres_sequence = []
  
  dict = {
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
    
    # save sequence of genres 
    genres_sequence.append(predicted_genre)
    
    dict[predicted_genre] += 1
    
  # extract main genre from the dictionary
  genre = get_key_with_max_value(dict)
  
  return genre, dict, genres_sequence