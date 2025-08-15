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
        self.feature_engineering_file_path = feature_engineering_file_path
        
        
       
        
    def start_batch_prediction(self):
        try:
            # Load the feature engineering pipeline path 
            with open(self.feature_engineering_file_path, 'rb') as f:
                feature_pipeline = pickle.load(f)

            # Load the data transformation pipeline path
            with open(self.transformer_file_path, 'rb') as f:
                processor = pickle.load(f)
            
            #Load the model sepratly
            # 
            model = load_model(file_path = self.model_file_path)
            
            # Create a feature engineering pipeline
            feature_engineering_pipeline = Pipeline([
                ("feature_engineering",feature_pipeline)
                ])
            
            df= pd.read_csv(self.input_file_path)
            df.to_csv("df_zomato_delivery_time_prdiction.csv")
            
            #Apply Feature engineering pipeline steps
            # 
            df = feature_engineering_pipeline.transform(df)
            df_transformed = pd.DataFrame(df_transformed, columns=df.columns)
            
            df.to_csv("feature_enigneering.csv")
            
            FEATURE_ENINGEERING_PATH = FEATURE_ENG 
            os.makedirs(FEATURE_ENINGEERING_PATH, exist_ok=True)
            file_path= os.path.join(FEATURE_ENINGEERING_PATH, 'batch_feature_eng.csv')
            
            df.to_csv(file_path, index= False)
            
            # Time taken
            df = df.drop('Time_taken (min)', axis = 1)
            df.to_csv("time_taken_droped.cv")
            
            transformerd_data = processor.transform(df)
            file_path = os.path. join(FEATURE_ENINGEERING_PATH,'processor.csv')
            predictions = model.predict(transformerd_data)
            df_prediction = pd.DataFrame (predictions, columns=['prediction' ])
            
            
            BATCH_PREDICTION_PATH= BATCH_PREDICTION
            os.makedirs(BATCH_PREDICTION_PATH, exist_ok=True) 
            csv_path = os.path.join(BATCH_PREDICTION_PATH, 'output.csv')
            df_prediction.to_csv(csv_path, index= False)
            logging.info(f"Batch prediction done")
        
        except Exception as e:
            CustomException(e, sys)