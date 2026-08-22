"""Tests for src/tools/weather.py.

The real Open-Meteo API is never called in these tests -- `requests.get` is
mocked so the tests are fast, deterministic, and don't depend on network
access or the weather actually being any particular value.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
import requests

from src.tools.weather import (
    LocationNotFoundError,
    WeatherServiceError,
    get_weather,
)

GEOCODE_RESPONSE = {
    "results": [
        {
            "name": "Hyderabad",
            "latitude": 17.385,
            "longitude": 78.4867,
            "country": "India",
        }
    ]
}

FORECAST_RESPONSE = {
    "current": {
        "time": "2026-08-22T10:00",
        "temperature_2m": 28.4,
        "wind_speed_10m": 12.3,
        "weather_code": 2,
    }
}


class _FakeResponse:
    """Minimal stand-in for requests.Response used across these tests."""

    def __init__(self, json_data: dict, status_code: int = 200):
        self._json_data = json_data
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")

    def json(self) -> dict:
        return self._json_data


def test_get_weather_valid_location_returns_structured_data():
    with patch("src.tools.weather.requests.get") as mock_get:
        mock_get.side_effect = [
            _FakeResponse(GEOCODE_RESPONSE),
            _FakeResponse(FORECAST_RESPONSE),
        ]

        result = get_weather("Hyderabad")

    assert result["location"] == "Hyderabad"
    assert result["country"] == "India"
    assert isinstance(result["temperature_c"], float)
    assert isinstance(result["windspeed_kmh"], float)
    assert result["condition"] == "partly cloudy"
    assert mock_get.call_count == 2


def test_get_weather_unknown_location_raises_location_not_found():
    with patch("src.tools.weather.requests.get") as mock_get:
        mock_get.return_value = _FakeResponse({"results": []})

        with pytest.raises(LocationNotFoundError):
            get_weather("Nowhereville")


def test_get_weather_geocoding_network_failure_raises_weather_service_error():
    with patch("src.tools.weather.requests.get") as mock_get:
        mock_get.side_effect = requests.ConnectionError("network down")

        with pytest.raises(WeatherServiceError):
            get_weather("Hyderabad")


def test_get_weather_forecast_http_error_raises_weather_service_error():
    with patch("src.tools.weather.requests.get") as mock_get:
        mock_get.side_effect = [
            _FakeResponse(GEOCODE_RESPONSE),
            _FakeResponse({}, status_code=500),
        ]

        with pytest.raises(WeatherServiceError):
            get_weather("Hyderabad")


def test_get_weather_malformed_forecast_response_raises_weather_service_error():
    with patch("src.tools.weather.requests.get") as mock_get:
        mock_get.side_effect = [
            _FakeResponse(GEOCODE_RESPONSE),
            _FakeResponse({"current": {"temperature_2m": "not-a-number"}}),
        ]

        with pytest.raises(WeatherServiceError):
            get_weather("Hyderabad")


def test_get_weather_rejects_empty_location():
    with pytest.raises(WeatherServiceError):
        get_weather("   ")


def test_get_weather_unknown_weather_code_falls_back_gracefully():
    forecast = {
        "current": {
            "time": "2026-08-22T10:00",
            "temperature_2m": 10.0,
            "wind_speed_10m": 5.0,
            "weather_code": 987,  # not in the WMO table
        }
    }
    with patch("src.tools.weather.requests.get") as mock_get:
        mock_get.side_effect = [_FakeResponse(GEOCODE_RESPONSE), _FakeResponse(forecast)]

        result = get_weather("Hyderabad")

    assert result["condition"] == "unknown conditions"
