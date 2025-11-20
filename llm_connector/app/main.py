import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
from .config import settings
from .llm_client import LLMClient

logging.basicConfig(
    level = settings.log_level,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

app = FastAPI(
    title = "LLM Connector API",
    description = "API to connect to various Large Language Models (LLMs)",
    version = settings.version
)

llm_client = LLMClient()


class LLMRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=2000)
    context: Optional[str] = Field(None, max_length=5000)
    max_tokens: int = Field(500, ge=50, le=2000)



class LLMResponse(BaseModel):

    response: str
    model: str
    status: str
    error: Optional[str] = None


@app.get("/")
async def root():
    return {
        "service": settings.service_name,
        "version": settings.version,
        "status": "running"
    }

@app.get("/health")
async def health():
    llm_available = llm_client.health_check()

    return {
        "service":"healthy" if llm_available else "degraded",
        "llm_available": llm_available,
        "provider": settings.llm_provider,
        "model": settings.llm_model
    }


@app.post("/chat", response_model=LLMResponse)
async def chat(request: LLMRequest):
    logger.info(f"Received chat request: {request.prompt[:50]} for model: {settings.llm_model}")

    try:
        result = llm_client.generate_response(
            prompt=request.prompt,
            context=request.context,
            max_tokens=request.max_tokens
        )

        logger.info(f"Generated response for model: {settings.llm_model}")
        return LLMResponse(**result)
    
    except Exception as e:
        logger.error(f"Error in /generate endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=settings.api_host,
        port=settings.api_port
    )

