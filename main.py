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
                    "description": "The city and state, e.g. San Francisco, CA"
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
        "description": "Get the current location if user does not provide a valid location",
        "parameters": {
            "type": "object",
            "properties": {}
        }
    }
]
You must always select one of the above tools and respond with only a JSON object matching the following schema inside a <json></json> tags:
{
    "tool": <name of the selected tool>,
    "tool_input": <parameters for the selected tool, matching the tool\'s JSON schema>,
    "explanation": <The explanation why you choosed this tool.>
}
'''
prompt = 'What is the current weather?' # LLM should response choose the get_current_location function with args: {}

body = json.dumps({
    'prompt': f'\n\n{system_prompt}\n\nHuman: {prompt}\n\nAssistant:<json>', # Use <json> xml tag to make llm response json only
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
# {
#   "tool": "get_current_location",
#   "tool_input": {},
#   "explanation": "Since the user did not provide a location, I will use the get_current_location tool to get their current location and pass that to the get_current_weather tool."
# }
