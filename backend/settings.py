# config/settings.py
import os

class Settings:
    APP_NAME: str = "LumenALYZE API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    MAX_FILE_SIZE: int = int(os.getenv("MAX_FILE_SIZE", 50 * 1024 * 1024))  # 50MB

settings = Settings()
