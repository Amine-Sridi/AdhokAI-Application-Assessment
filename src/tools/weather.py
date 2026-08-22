"""Weather tool implementation.

This module contains the *actual* work performed when the LLM decides to call
the ``get_weather`` tool. It knows nothing about LLMs, tool schemas, or
conversation state -- it just takes a location string and returns structured
weather data (or raises a well-defined exception).

Provider: Open-Meteo (https://open-meteo.com/), chosen because it requires no
API key for basic geocoding + forecast lookups, which keeps setup friction low
for this exercise.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, asdict
from typing import Any

import requests

logger = logging.getLogger(__name__)

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
REQUEST_TIMEOUT_SECONDS = 10

# WMO weather interpretation codes -> short human-readable description.
# https://open-meteo.com/en/docs (see "weather_code" / WMO Weather interpretation codes)
_WMO_CODE_DESCRIPTIONS = {
    0: "clear sky",
    1: "mainly clear",
    2: "partly cloudy",
    3: "overcast",
    45: "fog",
    48: "depositing rime fog",
    51: "light drizzle",
    53: "moderate drizzle",
    55: "dense drizzle",
    56: "light freezing drizzle",
    57: "dense freezing drizzle",
    61: "slight rain",
    63: "moderate rain",
    65: "heavy rain",
    66: "light freezing rain",
    67: "heavy freezing rain",
    71: "slight snow fall",
    73: "moderate snow fall",
    75: "heavy snow fall",
    77: "snow grains",
    80: "slight rain showers",
    81: "moderate rain showers",
    82: "violent rain showers",
    85: "slight snow showers",
    86: "heavy snow showers",
    95: "thunderstorm",
    96: "thunderstorm with slight hail",
    99: "thunderstorm with heavy hail",
}


class WeatherServiceError(Exception):
    """Raised when weather data cannot be retrieved for any reason."""


class LocationNotFoundError(WeatherServiceError):
    """Raised when the given location could not be geocoded."""


@dataclass(frozen=True)
class WeatherResult:
    """Structured weather data returned to the caller (and, ultimately, the LLM)."""

    location: str
    country: str | None
    latitude: float
    longitude: float
    temperature_c: float
    windspeed_kmh: float
    condition: str
    observed_at: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _geocode(location: str) -> tuple[float, float, str, str | None]:
    """Resolve a free-text location name to coordinates using Open-Meteo's geocoder."""
    try:
        response = requests.get(
            GEOCODING_URL,
            params={"name": location, "count": 1, "language": "en", "format": "json"},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException as exc:
        raise WeatherServiceError(f"Geocoding request failed: {exc}") from exc
    except ValueError as exc:  # invalid JSON
        raise WeatherServiceError("Geocoding service returned an invalid response") from exc

    results = payload.get("results")
    if not results:
        raise LocationNotFoundError(f"Could not find a location matching '{location}'")

    top = results[0]
    try:
        latitude = float(top["latitude"])
        longitude = float(top["longitude"])
        resolved_name = str(top.get("name", location))
        country = top.get("country")
    except (KeyError, TypeError, ValueError) as exc:
        raise WeatherServiceError("Geocoding service returned malformed data") from exc

    return latitude, longitude, resolved_name, country


def _fetch_current_conditions(latitude: float, longitude: float) -> dict[str, Any]:
    """Fetch current weather conditions for a set of coordinates."""
    try:
        response = requests.get(
            FORECAST_URL,
            params={
                "latitude": latitude,
                "longitude": longitude,
                "current": "temperature_2m,wind_speed_10m,weather_code",
                "timezone": "auto",
            },
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException as exc:
        raise WeatherServiceError(f"Forecast request failed: {exc}") from exc
    except ValueError as exc:
        raise WeatherServiceError("Forecast service returned an invalid response") from exc

    current = payload.get("current")
    if not isinstance(current, dict):
        raise WeatherServiceError("Forecast response did not include current conditions")

    return current


def get_weather(location: str) -> dict[str, Any]:
    """Get current weather conditions for a given location name.

    This is the function the tool registry dispatches to when the LLM emits a
    ``get_weather`` tool call. It performs geocoding (name -> coordinates) and
    then queries current conditions, returning a plain dict that is easy to
    JSON-serialize back to the LLM.

    Args:
        location: Free-text location, e.g. "Hyderabad" or "Paris, France".

    Raises:
        LocationNotFoundError: If the location cannot be resolved.
        WeatherServiceError: If any upstream request fails or returns
            unexpected data.
    """
    if not isinstance(location, str) or not location.strip():
        raise WeatherServiceError("A non-empty 'location' string is required")

    latitude, longitude, resolved_name, country = _geocode(location.strip())
    current = _fetch_current_conditions(latitude, longitude)

    try:
        temperature_c = float(current["temperature_2m"])
        windspeed_kmh = float(current["wind_speed_10m"])
        weather_code = int(current["weather_code"])
        observed_at = str(current["time"])
    except (KeyError, TypeError, ValueError) as exc:
        raise WeatherServiceError("Forecast response was missing expected fields") from exc

    condition = _WMO_CODE_DESCRIPTIONS.get(weather_code, "unknown conditions")

    result = WeatherResult(
        location=resolved_name,
        country=country,
        latitude=latitude,
        longitude=longitude,
        temperature_c=temperature_c,
        windspeed_kmh=windspeed_kmh,
        condition=condition,
        observed_at=observed_at,
    )
    logger.debug("Resolved weather for %s: %s", location, result)
    return result.to_dict()
