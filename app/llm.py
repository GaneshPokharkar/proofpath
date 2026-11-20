"""LLM access with schema-validated structured output.

Structured output works by forcing the model to call a single tool whose input schema is the
Pydantic model's JSON schema; the tool input is then validated by Pydantic. If validation fails,
the model gets the validation error and one more try, and a second failure raises.
"""

from typing import Protocol, TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    pass


class StructuredLLM(Protocol):
    model_name: str

    def extract(self, system: str, user: str, schema: type[T]) -> T: ...


class AnthropicLLM:
    def __init__(self, api_key: str, model: str, max_tokens: int = 2048):
        import anthropic  # imported lazily so tests and ingestion don't need the SDK configured

        self._client = anthropic.Anthropic(api_key=api_key)
        self.model_name = model
        self._max_tokens = max_tokens

    def extract(self, system: str, user: str, schema: type[T]) -> T:
        tool = {
            "name": "record",
            "description": f"Record the result as {schema.__name__}.",
            "input_schema": schema.model_json_schema(),
        }
        messages: list[dict] = [{"role": "user", "content": user}]
        for attempt in range(2):
            response = self._client.messages.create(
                model=self.model_name,
                max_tokens=self._max_tokens,
                system=system,
                tools=[tool],
                tool_choice={"type": "tool", "name": "record"},
                messages=messages,
            )
            call = next((b for b in response.content if b.type == "tool_use"), None)
            if call is None:
                raise LLMError(f"{self.model_name} returned no tool call (stop: {response.stop_reason})")
            try:
                return schema.model_validate(call.input)
            except ValidationError as exc:
                if attempt == 1:
                    raise LLMError(f"output failed validation twice: {exc}") from exc
                messages += [
                    {"role": "assistant", "content": response.content},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": call.id,
                                "is_error": True,
                                "content": f"Invalid input, fix and call record again:\n{exc}",
                            }
                        ],
                    },
                ]
        raise AssertionError("unreachable")
