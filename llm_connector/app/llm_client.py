import logging
import requests
from typing import Dict, Optional
from .config import settings

logger = logging.getLogger(__name__)

class LLMClient:

    def __init__(self):
        self.api_url = settings.llm_api_url
        self.model = settings.llm_model
        self.provider = settings.llm_provider

    
    def generate_response(self, prompt: str, context: Optional[str]=None, max_tokens: int = 500) -> Dict[str, str]:

        try:
            if self.provider == "ollama":
                return self._generate_ollama(prompt, context, max_tokens)
            else:
                raise ValueError(f"Unsupported LLM provider: {self.provider}")
        
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return {
                "response":" Sorry, an error occurred while generating the response.",
                "error": str(e),
                "status":"error"
            }
        
    def _generate_ollama(self, prompt: str, context: Optional[str], max_tokens: int) -> Dict[str, str]:

        full_prompt = prompt
        if context:
            full_prompt = f"Context: {context}\n\n Request: {prompt}"


        payload = {
            "model": self.model,
            "prompt": full_prompt,
            "stream":False,
            "options":{
                "num_predict": max_tokens,
                "temperature":0.7,
            }
        }

        response = requests.post(
            f"{self.api_url}/api/generate",
            json=payload,
            timeout=60
        )

        response.raise_for_status()
        result = response.json()

        return {
            "response": result.get("response", ""),
            "model": self.model,
            "status": "success"
        }
    

    def health_check(self)->bool:
        try:
            response = requests.get(
                f"{self.api_url}/api/tags",
                timeout=5
            )
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"Health check failed: {str(e)}")
            return False
        


        