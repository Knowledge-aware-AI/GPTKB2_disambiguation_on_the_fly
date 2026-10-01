import json
import os
import re

from openai import OpenAI

ROLES = ("elicitation", "disambiguation", "description")

THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)


class ResponsesBackend:

    def __init__(self, client: OpenAI):
        self.client = client

    def describe(self) -> str:
        return "OpenAI Responses API (model from the prompter)"

    def send(self, body: dict) -> dict:
        return self.client.responses.create(**body).model_dump(mode="json")


class ChatCompletionsBackend:
    """
    Config keys (all but base_url and model are optional):
        base_url           e.g. "http://localhost:8000/v1"
        model              model name as the server knows it, e.g. "Qwen/Qwen3-32B"
        api_key            most local servers ignore it; defaults to "EMPTY"
        api_key_env        name of an environment variable holding the key, so it does not have to be in the config file;
                           takes precedence over api_key
        merge_system_prompt  put the system prompt at the start of the user message instead of sending a system
                           message, for models whose chat template has no system role (default False)
        structured_output  send response_format for json_schema / json_object prompts (default True);
                           turn off for servers without guided decoding, the prompts ask for JSON anyway
        max_tokens         overrides the prompter's max_output_tokens, e.g. for reasoning models that need
                           room to think before answering
        temperature        overrides the prompter's temperature
        extra_body         passed through as is, e.g. {"chat_template_kwargs": {"enable_thinking": false}} for Qwen3 on vLLM
    """

    def __init__(self, base_url: str, model: str, api_key: str = "EMPTY", api_key_env: str = None,
                 merge_system_prompt: bool = False, structured_output: bool = True,
                 max_tokens: int = None, temperature: float = None, extra_body: dict = None,
                 timeout: float = 600, max_retries: int = 5):
        if api_key_env:
            api_key = os.environ.get(api_key_env)
            if not api_key:
                raise ValueError(f"Environment variable {api_key_env} for the {model} API key is not set")
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout, max_retries=max_retries)
        self.base_url = base_url
        self.model = model
        self.merge_system_prompt = merge_system_prompt
        self.structured_output = structured_output
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.extra_body = extra_body

    def describe(self) -> str:
        return f"Chat Completions at {self.base_url} (model {self.model})"

    def to_chat_request(self, body: dict) -> dict:
        messages = [
            {"role": "system" if item["role"] == "developer" else item["role"], "content": item["content"]}
            for item in body["input"]
        ]
        if self.merge_system_prompt:
            messages = self._merge_system_into_user(messages)
        request = {"model": self.model, "messages": messages}

        max_tokens = self.max_tokens if self.max_tokens is not None else body.get("max_output_tokens")
        if max_tokens is not None:
            request["max_tokens"] = max_tokens
        temperature = self.temperature if self.temperature is not None else body.get("temperature")
        if temperature is not None:
            request["temperature"] = temperature

        fmt = (body.get("text") or {}).get("format") or {}
        if self.structured_output and fmt.get("type") == "json_schema":
            request["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": fmt["name"], "schema": fmt["schema"], "strict": fmt.get("strict", False)},
            }
        elif self.structured_output and fmt.get("type") == "json_object":
            request["response_format"] = {"type": "json_object"}

        if self.extra_body:
            request["extra_body"] = self.extra_body
        return request

    @staticmethod
    def _merge_system_into_user(messages: list[dict]) -> list[dict]:
        system_text = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        merged = [dict(m) for m in messages if m["role"] != "system"]
        if not system_text:
            return merged
        for m in merged:
            if m["role"] == "user":
                m["content"] = f"{system_text}\n\n{m['content']}"
                return merged
        return [{"role": "user", "content": system_text}] + merged

    def send(self, body: dict) -> dict:
        resp = self.client.chat.completions.create(**self.to_chat_request(body))
        return chat_completion_to_responses_body(
            resp.choices[0].message.content, resp.model, resp.usage.model_dump() if resp.usage else None
        )


def chat_completion_to_responses_body(content: str, model: str = None, usage: dict = None) -> dict:
    text = THINK_BLOCK.sub("", content or "").strip()
    return {
        "model": model,
        "output": [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": text}]}],
        "usage": usage,
    }


def _check_roles(config: dict, kind: str):
    unknown = set(config) - set(ROLES)
    if unknown:
        raise ValueError(f"Unknown roles in {kind} backend config: {sorted(unknown)}, expected some of {ROLES}")


def build_single_backends(openai_client: OpenAI, config: dict = None, timeout: float = 600, max_retries: int = 5) -> dict:
    config = config or {}
    _check_roles(config, "single")
    backends = {role: ResponsesBackend(openai_client) for role in ROLES}
    for role, role_config in config.items():
        backends[role] = ChatCompletionsBackend(**role_config, timeout=timeout, max_retries=max_retries)
    return backends


class OpenAIBatchBackend:

    endpoint = "/v1/responses"

    def __init__(self, client: OpenAI, completion_window: str = "24h"):
        self.client = client
        self.completion_window = completion_window

    def describe(self) -> str:
        return f"OpenAI Batch API on {self.endpoint} (model from the prompter)"

    def to_batch_line(self, request: dict) -> dict:
        return request

    def upload(self, requests: list[dict], jsonl_path) -> object:
        with open(jsonl_path, "w") as f:
            for request in requests:
                f.write(json.dumps(self.to_batch_line(request)) + "\n")
        with open(jsonl_path, "rb") as f:
            return self.client.files.create(file=f, purpose="batch")

    def create_batch(self, input_file_id: str, metadata: dict) -> object:
        return self.client.batches.create(
            input_file_id=input_file_id,
            endpoint=self.endpoint,
            completion_window=self.completion_window,
            metadata=metadata,
        )

    def retrieve(self, batch_id: str) -> object:
        return self.client.batches.retrieve(batch_id)

    def download_output(self, openai_batch) -> bytes:
        return self.client.files.content(openai_batch.output_file_id).content


class ChatCompletionsBatchBackend(OpenAIBatchBackend):

    endpoint = "/v1/chat/completions"

    def __init__(self, completion_window: str = "24h", **chat_config):
        self.chat = ChatCompletionsBackend(**chat_config)
        super().__init__(self.chat.client, completion_window)

    def describe(self) -> str:
        return f"Batch API on {self.chat.base_url}{self.endpoint} (model {self.chat.model})"

    def to_batch_line(self, request: dict) -> dict:
        body = self.chat.to_chat_request(request["body"])
        body.update(body.pop("extra_body", None) or {})
        return {"custom_id": request["custom_id"], "method": "POST", "url": self.endpoint, "body": body}

    def normalize_output_line(self, line: dict) -> dict:
        response = line.get("response") or {}
        body = response.get("body") or {}
        if not body.get("choices"):
            return line
        response["body"] = chat_completion_to_responses_body(
            body["choices"][0]["message"].get("content"), body.get("model"), body.get("usage")
        )
        return line

    def download_output(self, openai_batch) -> bytes:
        content = super().download_output(openai_batch)
        lines = [json.loads(line) for line in content.decode("utf-8").splitlines() if line.strip()]
        return "\n".join(json.dumps(self.normalize_output_line(line)) for line in lines).encode("utf-8")


def build_batch_backends(openai_client: OpenAI, config: dict = None, timeout: float = 600, max_retries: int = 5) -> dict:
    config = config or {}
    _check_roles(config, "batch")
    backends = {role: OpenAIBatchBackend(openai_client) for role in ROLES}
    for role, role_config in config.items():
        backends[role] = ChatCompletionsBatchBackend(**role_config, timeout=timeout, max_retries=max_retries)
    return backends
