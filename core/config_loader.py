import yaml

def load_settings():
    """Load global configuration file settings.yaml"""
    with open("settings.yaml", "r") as f:
        return yaml.safe_load(f)