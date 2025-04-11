from openai import OpenAI, AsyncOpenAI
from pydantic import BaseModel
import tiktoken
from typing import List
import os
import dotenv

dotenv.load_dotenv()

text_model = "openrouter/optimus-alpha"

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
    system_prompt: str = None,
    response_format: BaseModel = None,
    model: str = text_model,
):
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

    kwargs = {"model": model, "messages": messages}

    if response_format is not None:
        schema = response_format.model_json_schema()
        # print("schema", schema)

        def process_schema(schema_dict):
            if schema_dict.get("type") not in ["object", "array"]:
                return schema_dict

            processed = {
                "type": schema_dict.get("type", "object"),
                "additionalProperties": False,
            }

            # Process definitions
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
    messages: List[dict], response_format: BaseModel = None, model: str = text_model
):
    """
    Make a LLM call with a list of messages instead of a prompt + system prompt

    ### Args:
        `messages` (`list[dict]`): The list of messages to send to the LLM.
        `response_format` (`BaseModel`, optional): Pydantic model for structured responses. Defaults to None.
        `model` (`str`, optional): Model identifier to use. Defaults to "quasar-alpha".
    """
    kwargs = {"model": model, "messages": messages}

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
        print("response", response)
        return response_format.parse_raw(response.choices[0].message.content)

    return client.chat.completions.create(**kwargs).choices[0].message.content


async def llm_call_messages_async(
    messages: List[dict], response_format: BaseModel = None, model: str = text_model
):
    """
    Make a LLM call with a list of messages instead of a prompt + system prompt

    ### Args:
        `messages` (`list[dict]`): The list of messages to send to the LLM.
        `response_format` (`BaseModel`, optional): Pydantic model for structured responses. Defaults to None.
        `model` (`str`, optional): Model identifier to use. Defaults to "quasar-alpha".
    """
    kwargs = {"model": model, "messages": messages}

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
        # print("response", response)
        return response_format.parse_raw(response.choices[0].message.content)

    response = await async_client.chat.completions.create(**kwargs)
    # print("response", response)
    return response.choices[0].message.content

def num_tokens_from_messages(messages: List[dict], model: str = text_model) -> int:
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
    class TestOutput(BaseModel):
        name: str
        value: int
        is_valid: bool

    class TestOutputList(BaseModel):
        tests: List[TestOutput]

    test_prompt = "Create a test output with name 'example', value 42, and is_valid True and another test output with name 'example2', value 43, and is_valid False"
    structured_response = llm_call(
        test_prompt,
        system_prompt="You are a helpful assistant that always returns valid JSON responses.",
        response_format=TestOutputList,
    )
    print(structured_response)
