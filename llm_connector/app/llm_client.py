import logging
import requests
from typing import Dict, Optional
from .config import settings
import time

logger = logging.getLogger(__name__)


class LLMClient:
    def __init__(self, max_retries: int = 3):
        self.api_url = settings.llm_api_url
        self.model = settings.llm_model
        self.provider = settings.llm_provider
        self.max_retries = max_retries
        self.retry_delay = 2
        
    def generate_response(
        self, 
        prompt: str, 
        context: Optional[str] = None,
        max_tokens: int = 500
    ) -> Dict[str, str]:
        try:
            if self.provider == "ollama":
                return self._generate_ollama(prompt, context, max_tokens)
            else:
                raise ValueError(f"Provider {self.provider} no soportado")
                
        except Exception as e:
            logger.error(f"Error generando respuesta: {str(e)}")
            return {
                "response": "Sorry, there was an error processing your request.",
                "error": str(e),
                "model": self.model,
                "status": "error"
            }
    
    def _generate_ollama(
        self, 
        prompt: str, 
        context: Optional[str], 
        max_tokens: int
    ) -> Dict[str, str]:

        full_prompt = prompt
        if context:
            full_prompt = f"Contexto: {context}\n\nPregunta: {prompt}"
        
   
        payload = {
            "model": self.model,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": 0.7
            }
        }
        

        for attempt in range(self.max_retries):
            try:
                logger.info(f"Intentando solicitud a Ollama (intento {attempt + 1}/{self.max_retries})")
                
                response = requests.post(
                    f"{self.api_url}/api/generate",
                    json=payload,
                    timeout=120
                )
                
                response.raise_for_status()
                result = response.json()
                
                return {
                    "response": result.get("response", ""),
                    "model": self.model,
                    "status": "success"
                }
                
            except requests.exceptions.ConnectionError as e:
                logger.warning(f"Connection error (attempt {attempt + 1}): {str(e)}")
                if attempt < self.max_retries - 1:
                    logger.info(f"Retrying in {self.retry_delay}s...")
                    time.sleep(self.retry_delay)
                else:
                    raise
                    
            except requests.exceptions.Timeout:
                logger.warning(f"Timeout in request (attempt {attempt + 1})")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    raise
    
    def health_check(self) -> bool:
        try:
            response = requests.get(
                f"{self.api_url}/api/tags",
                timeout=5
            )
            
            if response.status_code == 200:
                data = response.json()
                models = data.get("models", [])
                has_llama2 = any(m.get("name") == self.model for m in models)
                
                logger.info(f"LLM Health Check: OK. Models: {[m.get('name') for m in models]}")
                return has_llama2
            
            return False
            
        except Exception as e:
            logger.warning(f"Health check failed: {str(e)}")
            return False