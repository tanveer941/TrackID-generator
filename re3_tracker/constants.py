# Network Constants
CROP_SIZE = 227
CROP_PAD = 2
MAX_TRACK_LENGTH = 3
LSTM_SIZE = 512

import os.path
import sys

if getattr(sys, 'frozen', False):
    # application_path = os.path.dirname(sys.executable)
    LOG_DIR = os.path.join(os.path.dirname(sys.executable), 'logs')
elif __file__:
    LOG_DIR = os.path.join(os.path.dirname(__file__), 'logs')
GPU_ID = '0'

# Drawing constants
OUTPUT_WIDTH = 640
OUTPUT_HEIGHT = 480
PADDING = 2
