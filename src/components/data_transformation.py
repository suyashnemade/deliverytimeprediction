import os
import sys
from src.constants import *
from src.logger import logging
from src.exception import CustomException
from src.config.configuration import *
from src.utils import save_obj

from dataclasses import dataclass

import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OneHotEncoder,OrdinalEncoder
from sklearn.pipeline import Pipeline


#feature engg class

class Feature_engg(BaseEstimator,TransformerMixin):
    def __init__(self) -> None:
        logging.info("*********feature Engineering started ********")
        

    #formula in notebook file
    def distance_numpy(self, df, lat1, lon1,lat2, lon2 ):
        p = np.pi/180
        a = 0.5 - np.cos((df[lat2]-df[lat1])*p)/2 + np.cos(df[lat1]*p) * np.cos(df[lat2]*p) * (1-np.cos((df[lon2]-df[lon1])*p))/2
        df['distance'] = 12734 * np.arccos(np.sort(a))
        
        
       #fuc to transform data 
    def transform_data(self, df):
        try:
            df.drop(['ID'], axis = 1, inplace = True)

            self.distance_numpy(df, 'Restaurant_latitude',
                                'Restaurant_longitude',
                                'Delivery_location_latitude',
                                'Delivery_location_longitude')
            
            df.drop(['Delivery_person_ID', 'Restaurant_latitude','Restaurant_longitude',
                                'Delivery_location_latitude',
                                'Delivery_location_longitude',
                                'Order_Date','Time_Orderd','Time_Order_picked'], axis=1,inplace=True)
            
            logging.info("droping columns from our original dataset")
            
            return df

        except Exception as e:
            raise CustomException( e,sys)
        
    #fit func    
    def fit(self,X,y=None):
        return self
    
    #execute data transform
    def transform(self,X:pd.DataFrame,y=None):
        try:    
            transformed_df=self.transform_data(X)
                
            return transformed_df
        except Exception as e:
            raise CustomException(e,sys) from e

@dataclass
class DataTransformationConfig():
    #file paths
    processed_obj_file_path = PREPROCESSING_OBJ_FILE
    transformed_train_path = TRANSFORMED_TRAIN_FILE_PATH
    transformed_test_path = TRANSFORMED_TEST_FILE_PATH
    feature_engg_obj_path = FEATURE_ENGG_OBJ_FILE_PATH
    
    
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        

    def get_data_transformation_obj(self):
        #applyi
        try:
            #columns in train data
            Road_traffic_density = ['Low', 'Medium', 'High', 'Jam']
            Weather_conditions = ['Sunny', 'Cloudy', 'Fog', 'Sandstorms', 'Windy', 'Stormy']

            categorical_columns = ['Type_of_order','Type_of_vehicle','Festival','City']
            ordinal_encoder = ['Road_traffic_density', 'Weather_conditions']
            numerical_column=['Delivery_person_Age','Delivery_person_Ratings','Vehicle_condition',
                              'multiple_deliveries','distance']

            #numerical pipeline def
            numerical_pipeline =Pipeline(steps=[
                ('impute', SimpleImputer(strategy= 'constant', fill_value=0)),
                ('scaler', StandardScaler(with_mean=False))
            ])
            
            #categorical pipeline 
            categorical_pipeline = Pipeline(steps=[
                ('impute', SimpleImputer(strategy= 'most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown="ignore")),
                ('scaler', StandardScaler(with_mean=False))
            ])
            
            #ordinal pipeline
            ordinal_pipeline = Pipeline(steps=[
                ('impute', SimpleImputer(strategy= 'most_frequent')),
                ('ordinal', OrdinalEncoder(categories=[Road_traffic_density, Weather_conditions])),
                ('scaler', StandardScaler(with_mean=False))
            ])
            
            #passing pipline
            preprocessor = ColumnTransformer([
                ('numerical_pipeline',numerical_pipeline, numerical_column),
                ('ordinal_pipeline',ordinal_pipeline,ordinal_encoder),
                ('categorical_pipeline', categorical_pipeline, categorical_columns)
                
            ])
            logging.info("pipeline steps completed")

            return preprocessor

           
            
            
        
        
        except Exception as e:
            raise CustomException(e, sys)
        

    #class to pass feature engg
    def get_feature_engg_obj(self):
        try:
            feature_engg = Pipeline(steps=[('fe', Feature_engg())])
            return feature_engg
            
            
            
        except Exception as e:
            raise CustomException(e,sys)
        
    
    #initiating data tranformation
    def inititate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Optaining FE steps object")
            
            #for feature engg obj
            fe_obj = self.get_feature_engg_obj()
    
            train_df = fe_obj.fit_transform(train_df)
            test_df = fe_obj.transform(test_df)
            
            train_df = pd.DataFrame(train_df)
            test_df = pd.DataFrame(test_df)


            train_df.to_csv("train_data.csv")
            test_df.to_csv("test_data.csv")

            #for preprosessing obj
            processing_obj = self.get_data_transformation_obj()

            traget_columns_name = "Time_taken (min)"

            X_train = train_df.drop(columns = traget_columns_name, axis = 1)
            y_train = train_df[traget_columns_name]

            X_test = test_df.drop(columns = traget_columns_name, axis = 1)
            y_test = test_df[traget_columns_name]
            

            X_train = processing_obj.fit_transform(X_train)
            X_test = processing_obj.transform(X_test)

            train_arr = np.c_[X_train, np.array(y_train)]
            test_arr = np.c_[X_test, np.array(y_test)]

            df_train = pd.DataFrame(train_arr)
            df_test = pd.DataFrame(test_arr)

            os.makedirs(os.path.dirname(self.data_transformation_config.transformed_train_path), exist_ok=True)
            df_train.to_csv(self.data_transformation_config.transformed_train_path, index = False, header = True)

            os.makedirs(os.path.dirname(self.data_transformation_config.transformed_test_path), exist_ok=True)
            df_test.to_csv(self.data_transformation_config.transformed_test_path, index = False, header = True)


            #saving obj
            save_obj(file_path = self.data_transformation_config.processed_obj_file_path,
                     obj = processing_obj)

            save_obj(file_path = self.data_transformation_config.feature_engg_obj_path,
                     obj = fe_obj)
            
            return(train_arr,
                   test_arr,
                   self.data_transformation_config.processed_obj_file_path)
        
        except Exception as e:
            raise CustomException( e,sys)