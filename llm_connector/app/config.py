from pydantic import BaseSettings

class Setting(BaseSettings):

    service_name: str = "LLM Connector"
    version: str = "1.0.0"


    llm_provider: str = "ollama"
    llm_model: str = "llama2"
    llm_api_url: str = "http://ollama:11434"

    api_host: str = "0.0.0.0"
    api_port: int = 8001

    log_level: str = "INFO"  


    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Setting()