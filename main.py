from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from model.model import predict_electricity_usage, predict_gdp_growth, predict_population, predict_electrification, predict_population_growth, predict_gdp_total
from model.model import UnknownCountryError, supported_countries
from model.model import __version__ as model_version
import os
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Solaire Predictor API")

# Browser origins allowed to call the prediction endpoints. CORS matches on
# scheme+host+port only — a path like "/dashboard" never matches an Origin
# header and was doing nothing, so the entries here are bare origins.
origins = [
    # Solaire, on Coolify
    "https://solaire.afrotomation.com",
    # the predictor's own landing page
    "https://predictor.afrotomation.com",
    # local development
    "http://localhost",
    "http://localhost:3000",
    "http://localhost:8080",
    # the app's old Vercel deployment, kept until that host is retired
    "https://solaire-lemon.vercel.app",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



class UserInput(BaseModel):
    country_code: str
    year: str
    use_linear: bool

class PredictionOut(BaseModel):
    predicted_value: float

# The models are trained on 54 African countries only. Asking for anything
# else used to raise IndexError deep in pandas and return a bare 500; answer
# with the supported set instead so callers can correct themselves.
@app.exception_handler(UnknownCountryError)
def unknown_country_handler(request: Request, exc: UnknownCountryError):
    return JSONResponse(
        status_code=422,
        content={
            "detail": (
                f"No trained data for country code '{exc.country_code}'. "
                "This model covers 54 African countries — see GET /countries."
            ),
            "country_code": exc.country_code,
            "supported_count": len(supported_countries()),
        },
    )


@app.get("/api/health")
def health():
    return {"status": "ok", "model_version": model_version}


@app.get("/countries")
def countries():
    """The countries the models can answer for — what the demo form offers."""
    entries = supported_countries()
    return {"count": len(entries), "countries": list(entries)}

@app.post("/predict-electricity-usage", response_model=PredictionOut)
def predict(payload: UserInput):
    energy_demand = predict_electricity_usage(payload.country_code, payload.year, payload.use_linear)
    return {"predicted_value": energy_demand}

@app.post("/predict-gdp-total", response_model=PredictionOut)
def predict(payload: UserInput):
    gdp_total = predict_gdp_total(payload.country_code, payload.year, payload.use_linear)
    return {"predicted_value": gdp_total}

@app.post("/predict-gdp-growth", response_model=PredictionOut)
def predict(payload: UserInput):
    gdp_growth = predict_gdp_growth(payload.country_code, payload.year, payload.use_linear)
    return {"predicted_value": gdp_growth}

@app.post("/predict-population", response_model=PredictionOut)
def predict(payload: UserInput):
    population = predict_population(payload.country_code, payload.year, payload.use_linear)
    return {"predicted_value": population}

@app.post("/predict-population-growth", response_model=PredictionOut)
def predict(payload: UserInput):
    pop_growth = predict_population_growth(payload.country_code, payload.year, payload.use_linear)
    return {"predicted_value": pop_growth}

@app.post("/predict-electrification-rate", response_model=PredictionOut)
def predict(payload: UserInput):
    rate = predict_electrification(payload.country_code, payload.year, payload.use_linear)
    return {"predicted_value": rate}

# Serve the static landing page at "/" — mounted LAST so it doesn't shadow
# the predict-* and /api/* routes above. `html=True` makes / fall back to
# /public/index.html. Resolved relative to this file so the import works
# the same in dev (`uv run`) and in the Docker image (cwd = /app).
_PUBLIC_DIR = os.path.join(os.path.dirname(__file__), "public")
if os.path.isdir(_PUBLIC_DIR):
    app.mount("/", StaticFiles(directory=_PUBLIC_DIR, html=True), name="static")


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
