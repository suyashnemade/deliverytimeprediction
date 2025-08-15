from src.constants import *
from src.config.configuration import *

import os
import sys

import pandas as pd
import numpy as np

from src.logger import logging
from src.exception import CustomException
import pickle
from src.utils import load_model

from sklearn.pipeline import Pipeline



PREDICTION_FOLDER = "batch_prediction"

PREDICTION_CSV = "prediction_csv"

PREDICTION_FILE = "output.csv"

FEATURE_ENG_FOLDER= "feature_engg"

ROOT_DIR= os.getcwd()

BATCH_PREDICTION = os.path.join(ROOT_DIR, PREDICTION_FOLDER, PREDICTION_CSV)
FEATURE_ENG = os.path.join(ROOT_DIR, PREDICTION_FOLDER, FEATURE_ENG_FOLDER)

class batch_prediction:
    def __init__(self, input_file_path,
                 model_file_path,
                 transformer_file_path,
                 feature_engineering_file_path) -> None:
        
        self.input_file_path = input_file_path
        self.model_file_path = model_file_path
        self.transformer_file_path = transformer_file_path
        self.feture_engineering_file_path = feature_engineering_file_path
        
        
        
        
