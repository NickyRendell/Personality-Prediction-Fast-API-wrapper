# Local imports
import datetime

# Third party imports
from pydantic import BaseModel, Field

from ms.functions import get_model_response
from ms import app



model_name = "Predict your extraversion from text"
version = "v1.0.0"


# Input for data validation
class Input(BaseModel):
    sentence: str

   
# Ouput for data validation
class Output(BaseModel):
    label: str



@app.get('/info')
async def model_info():
    """Return model information, version, how to call"""
    return {
        "name": model_name,
        "version": version
    }


@app.get('/health')
async def service_health():
    """Return service health"""
    return {
        "ok"
    }


@app.post('/predict', response_model=Output)
async def model_predict(input: Input):
    """Predict with input"""
    response = get_model_response(input)
    return response