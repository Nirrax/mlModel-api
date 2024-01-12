from dependencies import *

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