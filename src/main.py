from fastapi import FastAPI

app = FastAPI()

@app.post('/')
async def genre_endpoint(request):
  return request