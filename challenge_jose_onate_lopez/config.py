
# app service
app_host = '0.0.0.0'
app_port = 5000
app_title = 'house-price-predictor-service'
app_description = """
Python service that is able to fit a predictor model and deliver predictions
and recommendations about the prices of houses in the US.
"""
num_workers = 2
# logging
log_level = 'info'
# predictor
model_path = 'data/house_price_model.pickle'
