
from fastapi import FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from house_price_predictor import PredictorModel
from pydantic import BaseModel
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.cors import CORSMiddleware
from typing import Dict, Union

import config as cfg
import logging
import uvicorn


# logging
logging.basicConfig(
    format='%(asctime)s | %(levelname)s | %(message)s',
    level=logging.INFO,
    handlers=[logging.StreamHandler()],
)

# start fastapi
app = FastAPI(
    title=cfg.app_title,
    description=cfg.app_description,
    # openapi_url=None,
    # docs_url=None,
    redoc_url=None,
)
# add origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# data models
class ModelInput(BaseModel):
    overall_quality: int
    neighborhood: str
    bath_area: float
    property_area: int
    garage_area: int
    garage_age: float
    house_age: int
    spaciousness: float
    remodel_age: int
    bsmt_area: int

class PredictionModel(BaseModel):
    prediction: float


class PerformanceData(BaseModel):
    rmse: float
    mae: float

class PerformanceModel(BaseModel):
    train: PerformanceData
    test: PerformanceData


# Init PredictorModel
predictor_model = PredictorModel(cfg.model_path)


# model performance
@app.get("/model_performance")
def get_model_performance(request: Request) -> PerformanceModel:
    client = f"{request.client.host}:{request.client.port}"
    logging.info(f"GET {request.url} from {client}")
    logging.info(f"200 response: {predictor_model.metrics}")
    response = predictor_model.metrics
    return response


# model prediction
@app.get("/predict")
def get_model_prediction(model_inputs: ModelInput, request: Request) -> PredictionModel:
    client = f"{request.client.host}:{request.client.port}"
    logging.info(f"GET {request.url} from {client}")
    # evaluate AD status
    predicted_price = predictor_model.predict(model_inputs.dict())
    response = PredictionModel(prediction=predicted_price)
    logging.info(f"200 response: {response}")
    return response


# Wrong endpoint
@app.exception_handler(StarletteHTTPException)
def http_exception_handler(request: Request, exc):
    if str(exc.detail) == 'Not Found':
        client = f"{request.client.host}:{request.client.port}"
        logging.error(f"GET {request.url} from {client}: 404 wrong endpoint")
        return JSONResponse(
            status_code=exc.status_code,
            content=jsonable_encoder({"error_message": 'wrong endpoint'}),
        )
    else:
        return JSONResponse(
            status_code=exc.status_code,
            content=jsonable_encoder({"error_message": exc.detail}),
        )


if __name__ == "__main__":
    # deactivate uvicorn loggers
    uvicorn_log_config = uvicorn.config.LOGGING_CONFIG
    del uvicorn_log_config["loggers"]
    logging.info(f'Starting {cfg.app_title} app')
    # run app
    uvicorn.run(
        "main:app",
        host=cfg.app_host,
        port=cfg.app_port,
        log_level=cfg.log_level,
        workers=cfg.num_workers,
        reload=True,
    )
