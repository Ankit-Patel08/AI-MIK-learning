
from openai import OpenAI   #import the OpenAI class from the openai module
from dotenv import load_dotenv  #import the load_dotenv function from the dotenv module
import os                     #import the os module to access environment variables
from Utilities.utilities import log  #import the utilities module from the Utilities package
from tools import *  #import all the functions from the tools.py file

load_dotenv()  #load the environment variables from the .env file

client = OpenAI(
    base_url=os.getenv("BASE_URL"),     # specify that which model to use for generating responses   command to run the ollama server: ollama serve --model llama2(qwen3:4b)
    api_key=os.getenv("API_KEY")                  # for authentication, you need to provide your OpenAI API key BUT FOR THE OLLAMA NO NEED TO PROVIDE API KEY but due to openai structure we have to follow the same structure and provide a dummy key for the API key parameter.
)

messages = []  # initialize an empty list to store the conversation history

MAX_STEPS = 5  # set the maximum number of steps for the model to take in generating responses


messages.append({
    "role": "system",
    "content": """
    You are a helpful AI assistant.

    You have access to tools for getting current time, weather, and location.

    IMPORTANT RULES:
    1. Answer general knowledge questions using your own knowledge.
    2. Do not call tools when they are not required.
    3. Use tools only when the user asks for information that requires them.
    4. After receiving tool results, use them to generate the final answer.
    5. Never say that you cannot answer a question just because you don't have a tool for it.
    6. Keep your answers concise and understandable.
    """
})

while True:             # infinite loop to keep the program running and allow for multiple interactions with the model
    user_input = input("User: ")       # get user input from the command line
    if user_input.lower() in ["exit", "quit", "bye", "goodbye"]:    # check if the user wants to exit the program
        break          # exit the loop and end the program

    log(f"Received user input ({len(user_input)} characters)")
    messages.append({"role": "user", "content": user_input})  # add the user's input to the conversation history


    # Streaming the response from the model
    log("Calling model with streaming enabled")

    response = client.chat.completions.create(
        model=os.getenv("MODEL"),
        messages=messages,
        stream=True
    )

    full_response = ""

    print("Assistant: ", end="", flush=True)

    for chunk in response:
        content = chunk.choices[0].delta.content

        if content:
            print(content, end="", flush=True)
            full_response += content

    print()

    messages.append({
        "role": "assistant",
        "content": full_response
    })  # add the complete assistant response to the conversation history
