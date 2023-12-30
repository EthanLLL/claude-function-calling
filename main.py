import boto3
import json

brt = boto3.client(service_name='bedrock-runtime')

system_prompt = '''
You have access to the following tools:
[
    {
        "name": "get_current_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city or state which is required."
                },
                "unit": {
                    "type": "string",
                    "enum": [
                        "celsius",
                        "fahrenheit"
                    ]
                }
            },
            "required": ["location"]
        }
    },
    {
        "name": "get_current_location",
        "description": "Use this tool to get the current location if user does not provide a location",
        "parameters": {
            "type": "object",
            "properties": {}
        }
    }
]
Select one of the above tools if needed and if tool needed, respond with only a JSON object matching the following schema.:
{
    "result": "tool_use",
    "tool": <name of the selected tool, leave blank if no tools needed>,
    "tool_input": <parameters for the selected tool, matching the tool\'s JSON schema, leave blank if no tools needed>,
    "explanation": <The explanation why you choosed this tool.>
}
If no further tools needed, response with only a JSON object matching the following schema:
{
    "result": "stop",
    "content": <Your response to the user.>,
    "explanation": <The explanation why you get the final answer.>
}
'''
user_prompt = 'What is the current weather?' # LLM should response choose the get_current_location function with args: {}

prompt_list = [
    # Uncomment the chat message
    f'\n\n{system_prompt}',
    f'\n\nHuman: {user_prompt}',
    '\n\nAssistant: Should use get_current_location tool with args: {}',
    '\n\nHuman: I have used the get_current_location tool and the result is: Guangzhou',
    '\n\nAssistant: Should use get_current_weather tool with args: {"location": "Guangzhou"}',
    '\n\nHuman: I have used the get_current_weather tool and the result is: Rainy and 7 degrees.',
    '\n\nAssistant: <json>' # Use <json> xml tag to force llm response json only
]

body = json.dumps({
    'prompt': ''.join(prompt_list),    
    'max_tokens_to_sample': 300,
    'temperature': 0,
    'top_p': 0.9,
    'stop_sequences': ['</json>']
})

modelId = 'anthropic.claude-v2'
accept = 'application/json'
contentType = 'application/json'

response = brt.invoke_model(body=body, modelId=modelId, accept=accept, contentType=contentType)

response_body = json.loads(response.get('body').read())

# text
completion = response_body.get('completion')
print(completion)
print(json.loads(completion))
