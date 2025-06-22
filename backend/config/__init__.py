"""
Configuration package for LumenALYZE
Contains application settings and chart configurations
"""

from .settings import settings
from .chart_config import CHART_THEMES, CHART_LAYOUTS, DEFAULT_CHART_CONFIG

__all__ = [
    'settings',
    'CHART_THEMES', 
    'CHART_LAYOUTS',
    'DEFAULT_CHART_CONFIG'
]
