import fastapi
from fastapi import HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, validator
from typing import List, Dict
from challenge.model import DelayModel
import pandas as pd

delay_model = DelayModel()


class Flight(BaseModel):
    """
    Represents a flight with its necessary attributes for prediction.
    """

    OPERA: str
    TIPOVUELO: str
    MES: int

    @validator("OPERA")
    def validate_opera(cls, value: str) -> str:
        """
        Validates that the 'OPERA' field is not empty.

        Args:
            value (str): Airline operator name.

        Returns:
            str: Validated operator name.

        Raises:
            ValueError: If the operator name is empty.
        """
        if not value.strip():
            raise ValueError("Invalid or missing 'OPERA'.")
        return value

    @validator("TIPOVUELO")
    def validate_tipo_vuelo(cls, value: str) -> str:
        """
        Validates that 'TIPOVUELO' is either 'N' or 'I'.

        Args:
            value (str): Flight type.

        Returns:
            str: Validated flight type.

        Raises:
            ValueError: If the flight type is invalid.
        """
        if value not in ["N", "I"]:
            raise ValueError("Invalid or missing 'TIPOVUELO'.")
        return value

    @validator("MES")
    def validate_mes(cls, value: int) -> int:
        """
        Validates that the 'MES' field is between 1 and 12.

        Args:
            value (int): Month value.

        Returns:
            int: Validated month value.

        Raises:
            ValueError: If the month value is outside the valid range.
        """
        if not (1 <= value <= 12):
            raise ValueError("Invalid or missing 'MES'. Must be between 1 and 12.")
        return value


class PredictionRequest(BaseModel):
    """
    Represents a batch of flights to predict delays for.
    """

    flights: List[Flight]


app = fastapi.FastAPI()


@app.get("/health", status_code=200)
async def get_health() -> Dict[str, str]:
    """
    Health check endpoint to verify the service is running.

    Returns:
        dict: A dictionary indicating the service status.
    """
    return {"status": "OK"}


@app.post("/predict", status_code=200)
async def post_predict(request: PredictionRequest) -> Dict[str, List[int]]:
    """
    Predicts flight delays using the DelayModel.

    Args:
        request (PredictionRequest): Input data containing flight information.

    Returns:
        dict: A dictionary containing the prediction results.

    Raises:
        HTTPException: If the model file is missing, input data is invalid, or an internal error occurs.
    """
    try:
        flights = pd.DataFrame([flight.dict() for flight in request.flights])
        for col in delay_model._features_cols:
            if col not in flights.columns:
                flights[col] = 0
        features = flights[delay_model._features_cols]
        predictions = delay_model.predict(features)
        return {"predict": predictions}
    except FileNotFoundError as fnf_exc:
        raise HTTPException(
            status_code=500,
            detail=f"Model file not found. Ensure the model is trained and saved: {str(fnf_exc)}",
        )
    except ValueError as val_exc:
        raise HTTPException(
            status_code=400, detail=f"Invalid input data: {str(val_exc)}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """
    Custom handler for request validation errors, ensuring a 400 status code is returned.

    Args:
        request (Request): The incoming request.
        exc (RequestValidationError): The validation error.

    Returns:
        JSONResponse: A JSON response with status code 400 and detailed error information.
    """
    errors = []
    for error in exc.errors():
        loc = error.get("loc", [])
        msg = error.get("msg", "Invalid value")
        typ = error.get("type", "unknown_error")
        errors.append({"location": loc, "message": msg, "type": typ})
    return JSONResponse(
        status_code=400,
        content={"detail": errors},
    )
