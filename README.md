# house-price-predictor-service
Python service that is able to fit a predictor model and deliver predictions and recommendations about the prices of houses in the US.

## To consider:
- the model used for prediction was trained using the notebook data_exploration.ipynb
- the model is saved in the folder data as pickle file: house_price_model.pickle
- the application service was implemented using FastAPI and server with uvicorn.
- the service can be configured using the config.py file. Here you can edit the host and port used to run the app.
- the endpoints in the app can be tested using the notebook test_service.ipynb
- automatically FastAPI enable the app documentation in the url [host]:[port]/docs

## Steps to start the service:
- create a virtual environment
- enable the previous virtual environment created
- install the requirements using the requirements.txt file: pip install -r requirements.txt
- run the app: python main.py
