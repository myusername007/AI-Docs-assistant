from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.llm.base import LLMClient, LLMResponse
from app.llm.factory import get_llm_client

router = APIRouter(prefix="/chat", tags=["chat"])


class ChatRequest(BaseModel):
    prompt: str
    system: str | None = None


@router.post("")
async def chat(
    body: ChatRequest,
    llm: LLMClient = Depends(get_llm_client),
) -> LLMResponse:
    return await llm.generate(body.prompt, system=body.system)