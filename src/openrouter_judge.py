import os
from dotenv import load_dotenv
from pydantic import BaseModel
from deepeval.models.base_model import DeepEvalBaseLLM
from langchain_openrouter import ChatOpenRouter

load_dotenv(override=True)


def _text(msg) -> str:
    c = msg.content
    return c if isinstance(c, str) else "".join(
        p.get("text", "") if isinstance(p, dict) else str(p) for p in c
    )


class OpenRouterJudge(DeepEvalBaseLLM):
    def __init__(self, model_name: str = "openai/gpt-oss-20b:free"):
        self.model_name = model_name
        self.model = ChatOpenRouter(model=model_name, temperature=0, max_retries=5)

    def load_model(self):
        return self.model

    @staticmethod
    def _json_hint(schema) -> str:
        return (
            "\n\nRespond with ONLY valid JSON that matches this JSON schema. "
            "No markdown fences, no extra text:\n"
            + str(schema.model_json_schema())
        )

    @staticmethod
    def _parse(text: str, schema):
        start, end = text.find("{"), text.rfind("}")
        return schema.model_validate_json(text[start:end + 1])

    def generate(self, prompt: str, schema: BaseModel | None = None):
        if schema is None:
            return _text(self.model.invoke(prompt))
        out = _text(self.model.invoke(prompt + self._json_hint(schema)))
        return self._parse(out, schema)

    async def a_generate(self, prompt: str, schema: BaseModel | None = None):
        if schema is None:
            return _text(await self.model.ainvoke(prompt))
        out = _text(await self.model.ainvoke(prompt + self._json_hint(schema)))
        return self._parse(out, schema)

    def get_model_name(self):
        return self.model_name