
from fastapi import FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from house_price_predictor import PredictorModel
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

# Init PredictorModel
predictor_model = PredictorModel(cfg.model_path)


# model performance
@app.get("/model_performance")
def get_model_performance(request: Request):
    client = f"{request.client.host}:{request.client.port}"
    logging.info(f"GET {request.url} from {client}")
    logging.info(f"200 response: {predictor_model.metrics}")
    return predictor_model.metrics


# model prediction
@app.get("/predict")
def get_model_prediction(model_inputs: Dict[str, Union[int, float, str]], request: Request):
    client = f"{request.client.host}:{request.client.port}"
    logging.info(f"GET {request.url} from {client}")
    # evaluate AD status
    response = {
        'prediction': predictor_model.predict(model_inputs),
    }
    logging.info(f"200 response: {response}")
    return response


# Missing Endpoint
@app.get("/")
def missing_endpoint(request: Request):
    client = f"{request.client.host}:{request.client.port}"
    logging.info(f"GET {request.url} from {client}")
    logging.error(f"404 missing endpoint")
    raise HTTPException(status_code=400, detail='missing endpoint')


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
