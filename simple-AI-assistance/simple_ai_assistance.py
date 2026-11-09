
from openai import OpenAI   #import the OpenAI class from the openai module
from dotenv import load_dotenv  #import the load_dotenv function from the dotenv module
import os                     #import the os module to access environment variables
from types import SimpleNamespace  # import SimpleNamespace to create objects from accumulated tool calls
from Utilities.utilities import log  #import the utilities module from the Utilities package
from tools import *  #import all the functions from the tools.py file

load_dotenv()  #load the environment variables from the .env file

client = OpenAI(
    base_url=os.getenv("BASE_URL"),     # specify that which model to use for generating responses   command to run the ollama server: ollama serve --model llama2(qwen3:4b)
    api_key=os.getenv("API_KEY")                  # for authentication, you need to provide your OpenAI API key BUT FOR THE OLLAMA NO NEED TO PROVIDE API KEY but due to openai structure we have to follow the same structure and provide a dummy key for the API key parameter.
)

messages = []  # initialize an empty list to store the conversation history

MAX_STEPS = 5  # set the maximum number of steps for the model to take in generating a response

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
    messages_before_turn = len(messages)  # store the current conversation history length
    messages.append({"role": "user", "content": user_input})  # add the user's input to the conversation history

    interrupted = False


    for step in range(MAX_STEPS):  # loop through the maximum number of steps to allow the model to generate a response

        log(f"Calling model, step {step + 1}/{MAX_STEPS}")

        response = client.chat.completions.create(
            model=os.getenv("MODEL"),
            messages=messages,
            tools=TOOLS,
            stream=True
        )

        full_response = ""
        tool_calls = {}

        print("Assistant: ", end="", flush=True)
        printed_content = False
        try:             # this try to handle the keyboard interrupt error when the user presses ctrl+c to stop the program
            for chunk in response:  # iterate over the streaming chunks received from the model

                delta = chunk.choices[0].delta

                if delta.content:  # check if the model has generated text content

                    print(delta.content, end="", flush=True)
                    full_response += delta.content
                    printed_content = True

                if delta.tool_calls:  # check if the model has generated any tool calls

                    for tool_call in delta.tool_calls:

                        index = tool_call.index  # get the index of the tool call

                        if index not in tool_calls:
                            tool_calls[index] = {
                                "id": "",
                                "type": "function",
                                "function": {
                                    "name": "",
                                    "arguments": ""
                                }
                            }

                        if tool_call.id:
                            tool_calls[index]["id"] = tool_call.id

                        if tool_call.function.name:
                            tool_calls[index]["function"]["name"] += tool_call.function.name

                        if tool_call.function.arguments:
                            tool_calls[index]["function"]["arguments"] += tool_call.function.arguments
        except KeyboardInterrupt:
           print("\nKeyboardInterrupt received. Stopping the model response stream.")
           response.close()
           tool_calls = {}  # clear the tool calls to prevent further processing
           full_response = ""  # clear the full response to prevent further processing
           interrupted = True
           break  # continue to the next iteration of the loop if an error occurs while processing the model response       

        if printed_content:
            print()

        if not tool_calls:  # check if the model has made any tool calls

            messages.append({
                "role": "assistant",
                "content": full_response
            })  # add the final assistant response to the conversation history

            if not printed_content:
                print("Assistant: No response generated.")

            break  # exit the loop if there are no tool calls

        tool_calls = dict(sorted(tool_calls.items()))

        messages.append({
            "role": "assistant",
            "content": full_response or None,
            "tool_calls": list(tool_calls.values())
        })  # add the assistant tool call to the conversation history

        for tool_call in tool_calls.values():  # iterate over the tool calls made by the model

            function_name = tool_call["function"]["name"]

            # convert the accumulated tool call into an object that execute_tool can understand
            tool_call_object = SimpleNamespace(
                id=tool_call["id"],
                function=SimpleNamespace(
                    name=function_name,
                    arguments=tool_call["function"]["arguments"]
                )
            )

            result = execute_tool(tool_call_object)  # call the execute_tool function to execute the tool call and get the result

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "content": str(result)
            })  # add the tool's response to the conversation history

        #till here we have called all the tools required by the model and added their results to the conversation history. Now we will make another call to the model to get its final response after having all the tool results in the conversation history.

    else:
        #Ran out of steps: The model never gave the final answer
        log(f"Model reached the maximum of {MAX_STEPS} steps without a final response")

        fallback_response = "I'm sorry, I couldn't complete the request within the allowed steps. Please try again."

        messages.append({
            "role": "assistant",
            "content": fallback_response
        })

        print("Assistant:", fallback_response)

    if interrupted:
        del messages[messages_before_turn:]
        log("Interrupted conversation turn removed from message history.")
