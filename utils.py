import os
import sys
import toml

def get_secrets_path():
    """Get the path to the secrets.toml file."""
    # For PyInstaller executable
    if getattr(sys, 'frozen', False):
        return os.path.join(os.path.dirname(sys.executable), "secrets.toml")
    # For running as a script
    return "secrets.toml"

def load_secrets():
    """Load secrets from secrets.toml."""
    path = get_secrets_path()
    if os.path.exists(path):
        return toml.load(path)
    return {}
