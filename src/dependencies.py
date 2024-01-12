import math
import numpy as np
import keras

from pydub import AudioSegment
import librosa

import base64

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware


