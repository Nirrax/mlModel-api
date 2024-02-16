from dependencies import *
from service import *
from schemas import Classification_request

app = FastAPI()

model = keras.models.load_model('CNN_Model.keras')

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

@app.post('/')
async def genre_endpoint(request: Classification_request):
  
  filename = ''
  filepath = '../mp3/'
  
  while True:
    
    filename = generate_unique_filename(request)
    
    if is_filename_unique(filename, filepath):
      filepath += filename
      break
  
  await save_file_from_request(request, filepath)
  convert_mp3_to_wav(filepath)
  mfccs = generate_mfcc_from_file(filepath)
  
  delete_wavs()
  
  filename = filename.replace(' ', '+')
  
  genre, genre_distribution, genre_sequence = predict(model, mfccs)
  
  response = {
    "genre": genre,
    "genreSequence": genre_sequence,
    "genreDistribution": genre_distribution,
    "fileName": filename,
  }
  
  return response

@app.get('/download/{filename}')
def download_mp3(filename: str):
  
  filename = filename.replace('+', ' ')
  filepath = '../mp3/' + filename +'.mp3'
  
  return FileResponse(path=filepath, filename=filepath, media_type='text/mp3')