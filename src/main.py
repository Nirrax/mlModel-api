from dependencies import *
from service import *

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
async def genre_endpoint(request: Request):
  
  filename = ''
  filepath = '../mp3/'
  
  data = await parse_request(request)
  
  while True:
    
    filename = generate_unique_filename(data)
    
    if is_filename_unique(filename, filepath):
      filepath += filename
      break
  
  await save_file_from_request(data, filepath)
  convert_mp3_to_wav(filepath)
  
  mfccs = generate_mfcc_from_file(filepath)
  
  delete_wavs()
  
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
  filename = filename[1:]
  
  filepath = '../mp3/' + filename +'.mp3'
  return FileResponse(path=filepath, filename=filepath, media_type='text/mp3')