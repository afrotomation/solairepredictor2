from functools import lru_cache
from pathlib import Path
import pandas as pd
import pickle

__version__ = "0.1.5"

BASE_DIR = Path(__file__).resolve(strict=True).parent


class UnknownCountryError(ValueError):
    """A country the models were never trained on.

    The training set is the 54 African countries in countries_mapping.csv.
    Anything else used to fall through to `.iloc[0]` on an empty frame and
    reach the caller as a 500 with a pandas traceback; the API turns this
    into a 422 that names the supported set instead.
    """

    def __init__(self, country_code):
        self.country_code = country_code
        super().__init__(f"No trained data for country code {country_code!r}")


@lru_cache(maxsize=None)
def load_model(modelname):
    """Unpickle a model once per process — every request used to re-read the
    pickle from disk, which dominated the response time."""
    with open(modelname, "rb") as saved_model:
        return pickle.load(saved_model)


@lru_cache(maxsize=1)
def _encoding_mapping():
    return pd.read_csv(BASE_DIR / "encoding_mapping.csv", encoding="utf-8")


@lru_cache(maxsize=1)
def _countries_mapping():
    return pd.read_csv(BASE_DIR / "countries_mapping.csv", encoding="utf-8")


@lru_cache(maxsize=1)
def supported_countries():
    """Every country the models can answer for, alphabetically by name."""
    countries = _countries_mapping().sort_values("Country Name")
    return tuple(
        {"code": row["Country Code"], "name": row["Country Name"]}
        for _, row in countries.iterrows()
    )


def _pick_model(metric, use_linear):
    variant = "linear" if use_linear else "boosting"
    label = "LINEAR MODEL" if use_linear else "GBR MODEL"
    return load_model(str(BASE_DIR / f"{variant}_{metric}-{__version__}.pkl")), label


def _features(country_code, year):
    """The one-hot feature row for a country/year, plus its display name.

    Returns a copy: the encoding frame is cached process-wide now, so
    inserting the Year column into a slice of it would corrupt the cache.
    """
    code = country_code.strip().upper()

    encoding = _encoding_mapping()
    encoded = encoding[encoding["Country Code"] == code]

    countries = _countries_mapping()
    names = countries.loc[countries["Country Code"] == code, "Country Name"]

    if encoded.empty or names.empty:
        raise UnknownCountryError(code)

    features = encoded.iloc[:, 1:].copy()
    features.insert(0, "Year", int(year))
    return features, names.iloc[0]


def predict_electricity_usage(country_code, year, use_linear=False):
    model, modeltype = _pick_model("electricity_usage", use_linear)
    features, country_name = _features(country_code, year)

    predicted_value = model.predict(features)
    adjusted_predicted_value_rmse = predicted_value * 1.2
    print(f'[{modeltype}] Predicted Energy Usage for {country_name} in {year}: {adjusted_predicted_value_rmse[0]:.2f}')
    return round(adjusted_predicted_value_rmse[0], 2)


def predict_gdp_growth(country_code, year, use_linear=False):
    model, modeltype = _pick_model("gdp_growth", use_linear)
    features, country_name = _features(country_code, year)

    predicted_value = model.predict(features)
    print(f'[{modeltype}] Predicted GDP Growth for {country_name} in {year}: {predicted_value[0]:.2f}%')
    return round(predicted_value[0], 2)


def predict_gdp_total(country_code, year, use_linear=False):
    model, modeltype = _pick_model("gdp_total", use_linear)
    features, country_name = _features(country_code, year)

    predicted_value = model.predict(features)
    print(f'[{modeltype}] Predicted GDP Total for {country_name} in {year}: {predicted_value[0]:.2f}')
    return round(predicted_value[0], 2)


def predict_population(country_code, year, use_linear=False):
    model, modeltype = _pick_model("population", use_linear)
    features, country_name = _features(country_code, year)

    predicted_value = model.predict(features)
    print(f'[{modeltype}] Predicted Population for {country_name} in {year}: {predicted_value[0]:.0f}')
    return round(predicted_value[0])


def predict_population_growth(country_code, year, use_linear=False):
    model, modeltype = _pick_model("population_growth", use_linear)
    features, country_name = _features(country_code, year)

    predicted_value = model.predict(features)
    print(f'[{modeltype}] Predicted Population Growth for {country_name} in {year}: {predicted_value[0]:.2f}%')
    return round(predicted_value[0], 2)


def predict_electrification(country_code, year, use_linear=False):
    model, modeltype = _pick_model("electrification_rate", use_linear)
    features, country_name = _features(country_code, year)

    predicted_value = model.predict(features)
    print(f'[{modeltype}] Predicted Electrification Rate for {country_name} in {year}: {predicted_value[0]:.2f}%')
    return round(predicted_value[0], 2)
