from dependencies import *

class Classification_request(BaseModel):
  fileName: str
  tags: dict
  base64Data: str