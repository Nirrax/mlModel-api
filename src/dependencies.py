import math
import os
import uuid
import numpy as np
import keras


from pydub import AudioSegment
import librosa
import music_tag

import base64

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel


