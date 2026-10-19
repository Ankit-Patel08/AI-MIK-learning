from openai import OpenAI   #import the OpenAI class from the openai module

client = OpenAI(
    base_url="http://127.0.0.1:11434/v1",     # specify that which model to use for generating responses   command to run the ollama server: ollama serve --model llama2(qwen3:4b)
    api_key="YOUR_API_KEY"                  # for authentication, you need to provide your OpenAI API key BUT FOR THE OLLAMA NO NEED TO PROVIDE API KEY but due to openai structure we have to follow the same structure and provide a dummy key for the API key parameter.
)

response = client.chat.completions.create(       # this is the openai endpoint for generating responses from the model. It takes in a list of messages and returns a response from the model.
    model="qwen3:4b",  # specify the model to use for generating responses
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello! Can you help me with some information?"}
    ]
)   
print("LLM response:")
print(response.choices[0].message.content)  # print the response from the model