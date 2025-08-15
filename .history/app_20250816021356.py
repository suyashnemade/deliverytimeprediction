from src.constants import *
from src.logger import logging
from src.exception import CustomException
import os
import sys
from src.config.configuration import *
from src.components.data_transformation import DataTransformation, DataTransformationConfig
from src.components.data_ingestion import DataIngestion, DataIngestionConfig
from src.components.model_trainer import ModelTrainer
from src.pipeline.prediction_pipeline import CustomData, PredictionPipeline
from src.pipeline.training_pipeline import Train
from Prediction.batch import batch_prediction
from werkzeug.utils import secure_filename
from flask import Flask, request, render_template

feature_engineering_file_path = FEATURE_ENGG_OBJ_FILE_PATH
transformer_file_path = PREPROCESSING_OBJ_FILE
model_file_path = MODEL_FILE_PATH
UPLOAD_FOLDER = 'batch_prediction/Uploaded_CSV_FILE'
ALLOWED_EXTENSIONS = {'csv'}

application = Flask(__name__, template_folder='templates')
app = application


@app.route('/')
def home_page():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('form.html')
    else:
        data = CustomData(
            Delivery_person_Age=int(request.form.get('Delivery_person_Age', 0)),
            Delivery_person_Ratings=float(request.form.get('Delivery_person_Ratings', 0.0)),
            Weather_conditions=request.form.get('Weather_conditions', ''),
            Road_traffic_density=request.form.get('Road_traffic_density', ''),
            Vehicle_condition=int(request.form.get('Vehicle_condition', 0)),
            multiple_deliveries=int(request.form.get('multiple_deliveries', 0)),
            distance=float(request.form.get('distance', 0.0)),
            Type_of_order=request.form.get('Type_of_order', ''),
            Type_of_vehicle=request.form.get('Type_of_vehicle', ''),
            Festival=request.form.get('Festival', ''),
            City=request.form.get('City', '')
        )

        final_new_data = data.get_data_as_dataframe()
        predict_pipeline = PredictionPipeline()
        pred = predict_pipeline.predict(final_new_data)
        result = int(pred[0])
        return render_template('form.html', final_result=result)


@app.route("/batch", methods=['GET', 'POST'])
def perform_batch_prediction():
    if request.method == 'GET':
        return render_template('batch.html')
    else:
        file = request.files.get('csv_file')  # Safely get file
        if file and file.filename:  # Ensure filename is not None
            filename = secure_filename(file.filename)
            if '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS:
                os.makedirs(UPLOAD_FOLDER, exist_ok=True)
                # Delete old files
                for f in os.listdir(UPLOAD_FOLDER):
                    file_path = os.path.join(UPLOAD_FOLDER, f)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                # Save new file
                file_path = os.path.join(UPLOAD_FOLDER, filename)
                file.save(file_path)
                logging.info("CSV received and uploaded")

                # Run batch prediction
                batch = batch_prediction(file_path, model_file_path, transformer_file_path, feature_engineering_file_path)
                batch.start_batch_prediction()

                return render_template("batch.html", prediction_result="Batch Prediction Done", prediction_type='batch')
            else:
                return render_template('batch.html', prediction_type='batch', error='Invalid file type')
        else:
            return render_template('batch.html', prediction_type='batch', error='No file uploaded')


@app.route('/train', methods=['GET', 'POST'])
def train():
    if request.method == 'GET':
        return render_template('train.html')
    else:
        try:
            pipeline = Train()
            pipeline.main()
            return render_template('train.html', message="Training complete")
        except Exception as e:
            logging.error(f"{e}")
            return render_template('index.html', error=str(e))


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=8888)
