
from datetime import datetime   
import requests

def get_current_time():     #this the tool which we are defining to get the current date and time. This function will be called by the model when it needs to get the current date and time.
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def get_city_weather(city):
    url = f"https://wttr.in/{city}?format=3"  # construct the URL for the weather API using the city name
    response = requests.get(url, timeout=10)  # make a GET request to the weather API
    response.raise_for_status()  # raise an exception if the request was unsuccessful
    return response.text


def get_my_city_location(city=None):
    try:
        if city:
            url = f"https://wttr.in/{city}?format=3"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                return response.text.strip()

        response = requests.get("https://ipinfo.io/json", timeout=10)
        if response.status_code != 200:
            return "Could not detect your current location."

        data = response.json()
        city_name = data.get("city") or "Unknown city"
        region = data.get("region") or ""
        country = data.get("country") or ""

        location_parts = [part for part in [city_name, region, country] if part]
        return ", ".join(location_parts) if location_parts else "Current location unavailable."

    except Exception as e:
        return f"Could not detect your current location: {e}"


TOOLS = [
    { 
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get the current date and time.",
            "parameters": {
                "type": "object",
                "properties": {},        # parameters goes here 
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_city_weather",
            "description": "Get the current weather for a given city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The name of the city to get the weather for."
                    }
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_my_city_location",
            "description": "Get the user's approximate current location based on their network IP. An optional city can be provided.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "Optional city name."
                    }
                },
                "required": []
            }
        }
    }
]

FUNCTIONS_NAME_TO_FUNCTION_MAPPING = {       # this is a dictionary that maps the function name to the actual function object. This allows us to call the function by its name when we receive a request from the model.
    "get_current_time": get_current_time,
    "get_city_weather": get_city_weather,
    "get_my_city_location": get_my_city_location
}
