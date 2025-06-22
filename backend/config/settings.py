import os
from typing import Optional

class Settings:
    """Application settings and configuration"""
    
    # Application settings
    APP_NAME: str = "LumenALYZE"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    
    # Server settings
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    
    # CORS settings
    CORS_ORIGINS: list = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://0.0.0.0:3000"
    ]
    
    # File upload settings
    MAX_FILE_SIZE: int = 50 * 1024 * 1024  # 50MB
    ALLOWED_FILE_TYPES: list = [".csv", ".xlsx", ".xls"]
    
    # Machine Learning settings
    DEFAULT_TEST_SIZE: float = 0.2
    DEFAULT_RANDOM_STATE: int = 42
    DEFAULT_N_CLUSTERS: int = 3
    DEFAULT_CONTAMINATION: float = 0.1
    
    # Logging settings
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE: str = "lumenalyze.log"
    
    # Export settings
    EXPORT_FORMATS: list = ["csv", "json", "excel"]
    MAX_EXPORT_ROWS: int = 100000

# Global settings instance
settings = Settings()
