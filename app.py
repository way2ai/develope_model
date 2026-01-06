import os
import json
import time
import uuid
from typing import Any, Iterator, List, Optional, Union

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

import infer

MODEL_DIR = r"E:\models\Tencent-Hunyuan\HY-MT1___5-1___8B-FP8"

app = FastAPI(title="Hunyuan MT (PyTorch)")


class TranslateReq(BaseModel):
    text: str
    max_new_tokens: int = 256


# @app.on_event("startup")
# def _startup():
#     infer.init(MODEL_DIR)


@app.get("/health")
def health():
    return infer.cuda_info()


@app.post("/translate")
def translate(req: TranslateReq):
    zh = infer.translate(req.text, max_new_tokens=req.max_new_tokens)
    return {"zh": zh}


def _sse(data: str, event: Optional[str] = None) -> str:
    if event:
        return f"event: {event}\ndata: {data}\n\n"
    return f"data: {data}\n\n"


@app.post("/translate/stream")
def translate_stream(req: TranslateReq):
    def gen() -> Iterator[bytes]:
        for piece in infer.stream_translate(req.text, max_new_tokens=req.max_new_tokens):
            yield _sse(piece).encode("utf-8")
        yield _sse("[DONE]", event="done").encode("utf-8")

    return StreamingResponse(gen(), media_type="text/event-stream")


# -------------------------
# OpenAI-compatible endpoint
# -------------------------

OPENAI_MODEL_ID = os.getenv("OPENAI_MODEL_ID", "hunyuan-mt")


class ChatMessage(BaseModel):
    role: str
    content: Union[str, List[Any]]  # 简化：若是列表则拼接为文本


class ChatCompletionRequest(BaseModel):
    model: str = OPENAI_MODEL_ID
    messages: List[ChatMessage]
    stream: bool = False
    max_tokens: Optional[int] = None  # 映射到 max_new_tokens


def _extract_last_user_text(messages: List[ChatMessage]) -> str:
    for m in reversed(messages):
        if m.role == "user":
            if isinstance(m.content, str):
                return m.content
            return " ".join(str(x) for x in m.content)
    m = messages[-1]
    return m.content if isinstance(m.content, str) else " ".join(str(x) for x in m.content)


@app.post("/v1/chat/completions")
def chat_completions(req: ChatCompletionRequest):
    text = _extract_last_user_text(req.messages)
    max_new_tokens = req.max_tokens or 256

    created = int(time.time())
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"

    if not req.stream:
        zh = infer.translate(text, max_new_tokens=max_new_tokens)
        return {
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": req.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": zh},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }

    def gen() -> Iterator[bytes]:
        # 首包：声明 assistant 角色
        first = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": req.model,
            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
        }
        yield _sse(json.dumps(first, ensure_ascii=False)).encode("utf-8")

        for piece in infer.stream_translate(text, max_new_tokens=max_new_tokens):
            chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": req.model,
                "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}],
            }
            yield _sse(json.dumps(chunk, ensure_ascii=False)).encode("utf-8")

        end_chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": req.model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        yield _sse(json.dumps(end_chunk, ensure_ascii=False)).encode("utf-8")
        yield _sse("[DONE]").encode("utf-8")

    return StreamingResponse(gen(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn

    infer.init(MODEL_DIR)
    uvicorn.run(app, host="0.0.0.0", port=8000)