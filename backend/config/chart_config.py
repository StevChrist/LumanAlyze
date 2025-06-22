CHART_THEMES = {
    'lumenalyze': {
        'colors': ['#014F39', '#FBF7C7', '#E8E8E6', '#120F0A'],
        'background': '#014F39',
        'grid_color': 'rgba(251, 247, 199, 0.2)',
        'text_color': '#FBF7C7',
        'font_family': 'Lora, serif'
    }
}

CHART_LAYOUTS = {
    'prediction': {
        'title': 'Prediction Analysis',
        'xaxis_title': 'Actual Values',
        'yaxis_title': 'Predicted Values',
        'showlegend': True
    },
    'anomaly': {
        'title': 'Anomaly Detection',
        'xaxis_title': 'Feature 1',
        'yaxis_title': 'Feature 2',
        'showlegend': True
    },
    'clustering': {
        'title': 'Data Clustering',
        'xaxis_title': 'Feature 1', 
        'yaxis_title': 'Feature 2',
        'showlegend': True
    }
}

DEFAULT_CHART_CONFIG = {
    'displayModeBar': True,
    'displaylogo': False,
    'responsive': True,
    'modeBarButtonsToRemove': ['pan2d', 'lasso2d', 'select2d']
}
