#!/usr/bin/env python3
import logging
import os # Import os module
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Load EPHERGENT SYSTEM PROMPT from prompts/ephergent-universe-prompt.md  ---
def load_system_prompt() -> str | None:
    """
    Loads the system prompt from a markdown file.

    Returns:
        str | None: The loaded system prompt, or None if not found/readable.
    """
    # Default path relative to this script's location
    default_prompt_path = Path(__file__).parent.parent / 'prompts' / 'ephergent_universe_prompt.md'
    # Get path from environment variable, using default if not set
    prompt_path_str = os.getenv('EPHERGENT_SYSTEM_PROMPT_PATH', str(default_prompt_path))
    prompt_path = Path(prompt_path_str)

    if not prompt_path.exists():
        logger.error(f"System prompt file not found at configured path: {prompt_path}")
        # Optionally, try the default path again if the env var path failed
        if prompt_path != default_prompt_path and default_prompt_path.exists():
            logger.warning(f"Falling back to default system prompt path: {default_prompt_path}")
            prompt_path = default_prompt_path
        else:
            return None # Return None if neither path works
    try:
        with open(prompt_path, 'r', encoding='utf-8') as file:
            system_prompt = file.read()
        return system_prompt
    except Exception as e:
        logger.error(f"Error reading system prompt file {prompt_path}: {e}")
        return None

if __name__ == '__main__':
    # Example of how to use it
    print("Testing system prompt loading...")
    prompt = load_system_prompt()
    if prompt:
        print("System prompt loaded successfully:")
        print(prompt[:200] + "...") # Print preview
    else:
        print("Failed to load system prompt.")
