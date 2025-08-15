from flask import Flask
from src.logger import logging
from src.exception import CustomException
import sys

app = Flask(__name__)

@app.route('/' , methods = ['GET', 'POST'])
def index():
    
    try:
        raise Exception("testin exception")
    
    except Exception as e:
        ML = CustomException(e, sys)
        logging.info(ML.error_message)
        logging.info("test log")
        return "welcome"


    
    
if __name__ == "__main__":
    host = "127.0.0.1"
    port = 5000
    print(f"Server running! Open: http://{host}:{port}")
    
    app.run(debug=True)