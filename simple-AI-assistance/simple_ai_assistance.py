
from openai import OpenAI   #import the OpenAI class from the openai module
from dotenv import load_dotenv  #import the load_dotenv function from the dotenv module
import os                     #import the os module to access environment variables

load_dotenv()  #load the environment variables from the .env file

client = OpenAI(
    base_url=os.getenv("BASE_URL"),     # specify that which model to use for generating responses   command to run the ollama server: ollama serve --model llama2(qwen3:4b)
    api_key=os.getenv("API_KEY")                  # for authentication, you need to provide your OpenAI API key BUT FOR THE OLLAMA NO NEED TO PROVIDE API KEY but due to openai structure we have to follow the same structure and provide a dummy key for the API key parameter.
)

messages = []  # initialize an empty list to store the conversation history

while True:             # infinite loop to keep the program running and allow for multiple interactions with the model
    user_input = input("User: ")       # get user input from the command line
    if user_input.lower() in ["exit", "quit", "bye", "goodbye"]:    # check if the user wants to exit the program
        break          # exit the loop and end the program

    messages.append({"role": "user", "content": user_input})  # add the user's input to the conversation history

    response = client.chat.completions.create(       # this is the openai endpoint for generating responses from the model. It takes in a list of messages and returns a response from the model.
        model=os.getenv("MODEL"),  # specify the model to use for generating responses
        messages=messages
    )

    messages.append({"role": "assistant", "content": response.choices[0].message.content})  # add the model's response to the conversation history

    print("LLM response:")
    print(response.choices[0].message.content)  # print the response from the model
  