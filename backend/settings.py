import os
from typing import List

class Settings:
    APP_NAME: str = "LumenALYZE API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    
    # Server settings
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", 8000))
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    # File upload settings
    MAX_FILE_SIZE: int = int(os.getenv("MAX_FILE_SIZE", 50 * 1024 * 1024))  # 50MB
    
    # CORS settings
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "https://your-frontend-app.vercel.app",  # Ganti dengan URL Vercel Anda
        "*"  # Untuk development
    ]

settings = Settings()
