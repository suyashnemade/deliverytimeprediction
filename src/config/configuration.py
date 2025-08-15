from src.constants import *
import sys
import os

ROOT_DIR = ROOT_DIR_KEY

#data ingestion related


DATASET_PATH = os.path.join(ROOT_DIR,DATA_DIR,DATA_DIR_KEY)

#raw path var
RAW_FILE_PATH = os.path.join(ROOT_DIR, ARTIFACT_DIR_KEY, DATA_INGESTION_KEY,
                             CURR_TIME_STAMP,DATA_INGESTION_RAW_DATA_DIR,
                             RAW_DATA_DIR_KEY)

#train path var
TRAIN_FILE_PATH =os.path.join(ROOT_DIR, ARTIFACT_DIR_KEY, DATA_INGESTION_KEY,
                             CURR_TIME_STAMP,DATA_INGESTION_INGESTED_DATA_DIR_KEY,
                             TRAIN_DATA_DIR_KEY)

#test path var
TEST_FILE_PATH =os.path.join(ROOT_DIR, ARTIFACT_DIR_KEY, DATA_INGESTION_KEY,
                             CURR_TIME_STAMP,DATA_INGESTION_INGESTED_DATA_DIR_KEY,
                             TEST_DATA_DIR_KEY)





#data transformation related


PREPROCESSING_OBJ_FILE = os.path.join(ROOT_DIR,ARTIFACT_DIR_KEY,DATA_TRANSFORMATION_ARTIFACT,
                                      DATA_PREPROCESSED_DIR,DATA_TRANSFORMATION_PROCESSING_OBJ)

TRANSFORMED_TRAIN_FILE_PATH= os.path.join(ROOT_DIR,ARTIFACT_DIR_KEY,DATA_TRANSFORMATION_ARTIFACT,
                                         DATA_TRANSFORMED_DIR,TRANSFORMED_TRAIN_DIR_KEY )

TRANSFORMED_TEST_FILE_PATH =os.path.join(ROOT_DIR,ARTIFACT_DIR_KEY,DATA_TRANSFORMATION_ARTIFACT,
                                         DATA_TRANSFORMED_DIR,TRANSFORMED_TEST_DIR_KEY )


#feature engg file

FEATURE_ENGG_OBJ_FILE_PATH = os.path.join(ROOT_DIR,ARTIFACT_DIR_KEY,
                                          DATA_TRANSFORMATION_ARTIFACT,DATA_PREPROCESSED_DIR,"feature_engg.pkl")



#model training

MODEL_FILE_PATH = os.path.join(ROOT_DIR,ARTIFACT_DIR_KEY,MODEL_TRAINER_KEY,MODEL_OBJECT)