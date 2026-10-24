from datetime import datetime   
import requests

def get_current_time():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def get_city_weather(city):
    url = f"https://wttr.in/{city}?format=3"  # construct the URL for the weather API using the city name
    response = requests.get(url, timeout=10)  # make a GET request to the weather API
    return response.text

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
}
]

FUNCTIONS_NAME_TO_FUNCTION_MAPPING = {       # this is a dictionary that maps the function name to the actual function object. This allows us to call the function by its name when we receive a request from the model.
    "get_current_time": get_current_time,
    "get_city_weather": get_city_weather
}