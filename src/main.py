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
  
  filename = 'output'
  
  await save_file_from_request(request, filename)
  convert_mp3_to_wav(filename, filename)
  
  mfccs = generate_mfcc_from_file(filename+'.wav')
  
  print(predict(model, mfccs))
 
  return 'success'