
from house_price_predictor import PredictorModel

import config as cfg
import requests


url = f"http://{cfg.app_host}:{cfg.app_port}/"
predictor_model = PredictorModel(cfg.model_path)
model_input = {
    'property_area': 2316,
    'house_age': 4,
    'house_style': '2Story',
    'neighborhood': 'Gilbert',
    'overall_quality': 6,
    'overall_condition': 5,
    'spaciousness': 269.0,
    'liv_lot_ratio': 0.0,
    'remodel_age': 8,
    'bath_area': 2.5,
    'bsmt_area': 1009,
    'garage_area': 461,
    'garage_age': 6.0,
    'has_2ndfloor': 1,
    'has_porch': 1,
    'has_pool': 0,
    'has_multiple_kitchen': 0,
}


def test_home_endpoint():
    response = requests.get(
        url=url,
    )
    assert response.status_code == 404


def test_wrong_endpoint():
    response = requests.get(
        url=f"{url}wrong_endpoint",
    )
    assert response.status_code == 404


def test_wrong_endpoint_message():
    response = requests.get(
        url=f"{url}wrong_endpoint",
    )
    assert response.json() == {'error_message': 'wrong endpoint'}


def test_model_performance_status_code():
    response = requests.get(
        url=f"{url}model_performance",
    )
    assert response.status_code == 200


def test_model_performance_content():
    response = requests.get(
        url=f"{url}model_performance",
    )
    assert response.json() == predictor_model.metrics


def test_prediction_status_code():
    response = requests.get(
        url=f"{url}predict",
        json=model_input,
    )
    assert response.status_code == 200


def test_prediction_value():
    response = requests.get(
        url=f"{url}predict",
        json=model_input,
    )
    assert response.json() == {'prediction': 170124.437}


def test_prediction_missing_field():
    del model_input['property_area']
    response = requests.get(
        url=f"{url}predict",
        json=model_input,
    )
    assert response.status_code == 422


def test_prediction_wrong_dtype():
    model_input['property_area'] = 'any_string'
    response = requests.get(
        url=f"{url}predict",
        json=model_input,
    )
    assert response.status_code == 422
