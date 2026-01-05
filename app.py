import os
from typing import Iterator, Optional

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

import infer

MODEL_DIR = r"E:\models\Tencent-Hunyuan\HY-MT1___5-1___8B-FP8"

app = FastAPI(title="Hunyuan MT (PyTorch)")


class TranslateReq(BaseModel):
    text: str
    max_new_tokens: int = 256


@app.on_event("startup")
def _startup():
    infer.init(MODEL_DIR)


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