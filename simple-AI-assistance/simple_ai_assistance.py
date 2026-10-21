
import json

from openai import OpenAI   #import the OpenAI class from the openai module
from dotenv import load_dotenv  #import the load_dotenv function from the dotenv module
import os                     #import the os module to access environment variables
from tools import *  #import all the functions from the tools.py file

load_dotenv()  #load the environment variables from the .env file

client = OpenAI(
    base_url=os.getenv("BASE_URL"),     # specify that which model to use for generating responses   command to run the ollama server: ollama serve --model llama2(qwen3:4b)
    api_key=os.getenv("API_KEY")                  # for authentication, you need to provide your OpenAI API key BUT FOR THE OLLAMA NO NEED TO PROVIDE API KEY but due to openai structure we have to follow the same structure and provide a dummy key for the API key parameter.
)

messages = []  # initialize an empty list to store the conversation history

messages.append({"role": "system", "content": "Act like a very humble agent and respond always under 30 words"})  # add a system message to the conversation history to set the context for the model, giving it a personality 

while True:             # infinite loop to keep the program running and allow for multiple interactions with the model
    user_input = input("User: ")       # get user input from the command line
    if user_input.lower() in ["exit", "quit", "bye", "goodbye"]:    # check if the user wants to exit the program
        break          # exit the loop and end the program

    messages.append({"role": "user", "content": user_input})  # add the user's input to the conversation history

    response = client.chat.completions.create(       # this is the openai endpoint for generating responses from the model. It takes in a list of messages and returns a response from the model.
        model=os.getenv("MODEL"),  # specify the model to use for generating responses
        messages=messages,
        tools=TOOLS,  # pass the list of tools to the model so that it can use them to generate responses
    )

    model_response = response.choices[0].message  # get the model's response from the response object
    if model_response.tool_calls:
        messages.append(model_response)  # add the model's response to the conversation history

        for tool_call in model_response.tool_calls:  # iterate over the tool calls made by the model
            function_name = tool_call.function.name  # get the name of the tool(function) that was called
            function_args = json.loads(tool_call.function.arguments or "{}")  # get the arguments passed to the tool(function) that was called and convert them from a JSON string to a Python dictionary

            fn = FUNCTIONS_NAME_TO_FUNCTION_MAPPING.get(function_name)  # get the actual function object from the mapping dictionary using the function name
            result = fn(**function_args)  # call the function with the arguments and store the result

            messages.append({
                "role": "tool", 
                "tool_call_id": tool_call.id,
                "content": str(result)
                })  # add the tool's response to the conversation history

        #2nd call to the model after the tool has been called and the result has been added to the conversation history 
        client_response = client.chat.completions.create(
            model=os.getenv("MODEL"),  # specify the model to use for generating responses
            messages=messages
        )

        model_response = client_response.choices[0].message  # get the model's response from the response object

    messages.append({
        "role": "assistant", 
        "content": model_response.content
    })  # add the model's response to the conversation history  

    print("Assistant:", model_response.content)  # print the model's response to the command line

