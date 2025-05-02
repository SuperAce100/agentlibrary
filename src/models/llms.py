import json
from openai import OpenAI, AsyncOpenAI, ChatCompletion
from pydantic import BaseModel
import tiktoken
from typing import Any
import os
import dotenv
from models.tools import Tool, browser_tool, terminal_tool

dotenv.load_dotenv()

text_model = "openai/gpt-4.1-mini"

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

async_client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)


def llm_call(
    prompt: str,
    system_prompt: str | None = None,
    response_format: BaseModel | None = None,
    model: str = text_model,
) -> str | BaseModel:
    """
    Make a LLM call

    ### Args:
        `prompt` (`str`): The user prompt to send to the LLM.
        `system_prompt` (`str`, optional): System-level instructions for the LLM. Defaults to None.
        `response_format` (`BaseModel`, optional): Pydantic model for structured responses. Defaults to None.
        `model` (`str`, optional): Model identifier to use. Defaults to "gpt-4o-mini".

    ### Returns:
        The LLM's response, either as raw text or as a parsed object according to `response_format`.
    """
    messages = [
        {"role": "system", "content": system_prompt} if system_prompt else None,
        {"role": "user", "content": prompt},
    ]
    messages = [msg for msg in messages if msg is not None]

    kwargs: dict[str, Any] = {"model": model, "messages": messages}

    if response_format is not None:
        schema = response_format.model_json_schema()
        # print("schema", schema)

        def process_schema(schema_dict: dict[str, Any]) -> dict[str, Any]:
            if schema_dict.get("type") not in ["object", "array"]:
                return schema_dict

            processed = {
                "type": schema_dict.get("type", "object"),
                "additionalProperties": False,
            }

            if "$defs" in schema_dict:
                processed["$defs"] = {}
                for def_name, def_schema in schema_dict["$defs"].items():
                    processed["$defs"][def_name] = process_schema(def_schema)

            if "required" in schema_dict:
                processed["required"] = schema_dict["required"]

            if "title" in schema_dict:
                processed["title"] = schema_dict["title"]

            if "properties" in schema_dict:
                processed["properties"] = {}
                for prop_name, prop_schema in schema_dict["properties"].items():
                    processed["properties"][prop_name] = process_schema(prop_schema)

            if "items" in schema_dict:
                processed["items"] = process_schema(schema_dict["items"])

            return processed

        processed_schema = process_schema(schema)

        kwargs["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": response_format.__name__,
                "strict": True,
                "schema": processed_schema,
            },
        }

        response = client.chat.completions.create(**kwargs)

        if not response.choices or not response.choices[0].message.content:
            raise ValueError(
                "No valid response content received from the API", response
            )

        try:
            return response_format.model_validate_json(
                response.choices[0].message.content
            )
        except Exception as e:
            print("Failed to parse response:", response.choices[0].message.content)
            raise ValueError(f"Failed to parse response: {e}")

    return client.chat.completions.create(**kwargs).choices[0].message.content


def llm_call_messages(
    messages: list[dict[str, str]],
    response_format: BaseModel = None,
    model: str = text_model,
) -> str | BaseModel:
    """
    Make a LLM call with a list of messages instead of a prompt + system prompt

    ### Args:
        `messages` (`list[dict]`): The list of messages to send to the LLM.
        `response_format` (`BaseModel`, optional): Pydantic model for structured responses. Defaults to None.
        `model` (`str`, optional): Model identifier to use. Defaults to "quasar-alpha".
    """
    kwargs: dict[str, Any] = {"model": model, "messages": messages}

    if response_format is not None:
        schema = response_format.schema()
        kwargs["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": response_format.__name__,
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": schema["properties"],
                    "required": schema["required"],
                    "additionalProperties": False,
                },
            },
        }

        response = client.chat.completions.create(**kwargs)
        try:
            return response_format.parse_raw(response.choices[0].message.content)
        except Exception as e:
            print("Failed to parse response:", response)
            raise ValueError(f"Failed to parse response: {e}")

    response = client.chat.completions.create(**kwargs)
    try:
        return response.choices[0].message.content
    except Exception as e:
        print("Failed to parse response:", response)
        raise ValueError(f"Failed to parse response: {e}")


def _llm_call_tools(
    msgs: list[dict[str, str]], tools: list[Tool], model: str = text_model
) -> Any:
    """
    Simple LLM call with tools. No structured response.
    """
    resp = client.chat.completions.create(
        model=model, tools=[tool.to_openai_tool() for tool in tools], messages=msgs
    )
    try:
        msgs.append(resp.choices[0].message.model_dump())
    except Exception as e:
        raise ValueError(f"Failed to parse response: {e}, {resp}")

    return resp


def _get_tool_responses(
    response: ChatCompletion, tools: list[Tool]
) -> list[dict[str, Any]]:
    MAX_TOOL_RESP_LENGTH = 10000
    tool_responses = []

    for tool_call in response.choices[0].message.tool_calls:
        tool_name = tool_call.function.name
        tool_args = json.loads(tool_call.function.arguments)
        chosen_tool = [tool for tool in tools if tool.name == tool_name][0]
        tool_result = chosen_tool(**tool_args)

        if len(tool_result) > MAX_TOOL_RESP_LENGTH:
            tool_result = tool_result[:MAX_TOOL_RESP_LENGTH] + "..."

        tool_responses.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": tool_name,
                "content": tool_result,
            }
        )

    return tool_responses


def llm_call_with_tools(
    messages: list[dict[str, str]], tools: list[Tool], model: str = text_model
) -> str:
    while True:
        resp = _llm_call_tools(messages, tools, model)
        if resp.choices[0].message.tool_calls is not None:
            messages.extend(_get_tool_responses(resp, tools))
        else:
            break
    return messages[-1]["content"]


async def llm_call_messages_async(
    messages: list[dict[str, str]],
    response_format: BaseModel = None,
    model: str = text_model,
) -> str | BaseModel:
    """
    Make a LLM call with a list of messages instead of a prompt + system prompt

    ### Args:
        `messages` (`list[dict]`): The list of messages to send to the LLM.
        `response_format` (`BaseModel`, optional): Pydantic model for structured responses. Defaults to None.
        `model` (`str`, optional): Model identifier to use. Defaults to "quasar-alpha".
    """
    kwargs: dict[str, Any] = {"model": model, "messages": messages}

    if response_format is not None:
        schema = response_format.schema()
        kwargs["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": response_format.__name__,
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": schema["properties"],
                    "required": schema["required"],
                    "additionalProperties": False,
                },
            },
        }

        response = await async_client.chat.completions.create(**kwargs)
        try:
            return response_format.parse_raw(response.choices[0].message.content)
        except Exception as e:
            print("Failed to parse response:", response)
            raise ValueError(f"Failed to parse response: {e}")

    response = await async_client.chat.completions.create(**kwargs)
    try:
        return response.choices[0].message.content
    except Exception as e:
        print("Failed to parse response:", response)
        raise ValueError(f"Failed to parse response: {e}")


def num_tokens_from_messages(
    messages: list[dict[str, str]], model: str = text_model
) -> int:
    """Returns the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("o200k_base")

    num_tokens = 0
    for message in messages:
        num_tokens += 4
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += -1
    num_tokens += 2
    return num_tokens


if __name__ == "__main__":
    # prompt = "What is the capital of the moon?"
    # response = llm_call(prompt)
    # print(response)

    # messages = [
    #     {"role": "user", "content": prompt}
    # ]
    # response = llm_call_messages(messages)
    # print(response)

    # class TestOutput(BaseModel):
    #     name: str
    #     value: int
    #     is_valid: bool

    # class TestOutputList(BaseModel):
    #     tests: List[TestOutput]

    # test_prompt = "Create a test output with name 'example', value 42, and is_valid True and another test output with name 'example2', value 43, and is_valid False"
    # structured_response = llm_call(
    #     test_prompt,
    #     system_prompt="You are a helpful assistant that always returns valid JSON responses.",
    #     response_format=TestOutputList,
    # )
    # print(structured_response)

    def get_weather(city: str) -> str:
        if city == "Stanford":
            return "The weather in Stanford is sunny."
        else:
            return f"The weather in {city} is cloudy."

    class WeatherArgs(BaseModel):
        city: str

    def get_news(topic: str) -> str:
        if topic == "Stanford":
            return "Stanford is doing really great research in AI."
        else:
            return f"The news about {topic} is that it is good."

    class NewsArgs(BaseModel):
        topic: str

    weather_tool = Tool(
        name="get_weather",
        description="Get the weather in a city",
        function=get_weather,
        argument_schema=WeatherArgs,
    )

    news_tool = Tool(
        name="get_news",
        description="Get the news about a topic",
        function=get_news,
        argument_schema=NewsArgs,
    )

    tools = [browser_tool, terminal_tool]
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant that can use tools to answer questions.",
        },
        {"role": "user", "content": "Tell me about the weather in Stanford"},
    ]

    response = llm_call_with_tools(messages, tools)
    print(response)

    # print("\n".join([f"{msg['role']}: {msg['content']}" for msg in messages]))
