import os
import sys
from src.constants import *
from src.logger import logging
from src.exception import CustomException
from src.config.configuration import *
from src.utils import save_obj
from src.utils import evaluate_model

from dataclasses import dataclass

import pandas as pd
import numpy as np


from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRFRegressor



@dataclass
class ModelTrainerConfig:
    trained_model_file_path = MODEL_FILE_PATH
    
    

class ModelTrainer:
    def init (self):
        self.model_trainer_config = ModelTrainerConfig
        
    def inititate_model_traning(self, train_array, test_array):
        try:
            X_train, y_train, X_test, y_test = (train_array[:, :-1], train_array[:, -1],
                                                test_array[:, :-1], test_array[:, -1])
            
            models = {"XGBRegressor": XGBRFRegressor(),
                     "DecisionTreeRegressor": DecisionTreeRegressor(),
                     "GradientBoostingRegressor": GradientBoostingRegressor(),
                     "RandomForestRegressor": RandomForestRegressor(),
                     "SVR": SVR()
                     }
                     
                     
                     
            model_report: dict = evaluate_model(X_train, y_train,X_test, y_test, models)
            print()

            best_model_score= max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model= models[best_model_name]

            print (f"Best Model Found, Hodel Name: (best model_name), R2 Score: (best model_score)") 
            logging.info(f"Best Model Found, Model Name: pest_model_neme). R2 Score: (best model_score)")
            
            
            save_obj(file_path= self.model_trainer_config.trained_model_file_path
                     obj = best_model)
                            
        except Exception as e:
            raise CustomException(e, sys)