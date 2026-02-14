
import torch
import numpy as np
from loguru import logger
from typing import List, Union, Dict, Tuple, Optional
import threading
import json
import os
import warnings
import logging
from datetime import datetime
from pathlib import Path
import chromadb
from config import EMBEDDING_DIMENSION

# Suppress NVML warnings (expected in non-GPU environments)
warnings.filterwarnings('ignore', message='Can\'t initialize NVML')

# Disable ChromaDB telemetry to prevent telemetry event errors
os.environ['CHROMA_TELEMETRY_DISABLED'] = '1'

# Suppress ChromaDB telemetry warnings
logging.getLogger('chromadb').setLevel(logging.ERROR)

