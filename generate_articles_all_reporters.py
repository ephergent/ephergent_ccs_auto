#!/usr/bin/env python3
import json
import subprocess
import sys
from pathlib import Path
import logging

# Set up logging for this script
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define the path to the prompts file and the app script
# Assumes this script is in the same directory as app.py and the prompts directory
PROMPTS_FILE = Path(__file__).parent / 'prompts' / 'personality_prompts.json'
APP_SCRIPT = Path(__file__).parent / 'app.py'

def generate_articles_for_all_reporters():
    """
    Reads reporter IDs from prompts/personality_prompts.json and runs app.py
    to auto-generate an article for each, skipping all steps except generation.
    """
    if not PROMPTS_FILE.exists():
        logger.error(f"Error: Prompts file not found at {PROMPTS_FILE}")
        sys.exit(1)

    if not APP_SCRIPT.exists():
        logger.error(f"Error: app.py script not found at {APP_SCRIPT}")
        sys.exit(1)

    try:
        with open(PROMPTS_FILE, 'r', encoding='utf-8') as f:
            prompts_data = json.load(f)
        reporters = prompts_data.get('reporters', [])

        if not reporters:
            logger.info("No reporters found in the prompts file.")
            sys.exit(0)

        logger.info(f"Found {len(reporters)} reporters. Starting article generation for each.")

        # Define the steps to skip as per the user's request
        skip_steps = "audio,git,youtube,archive,export,image,video,social"
        logger.info(f"Skipping steps: {skip_steps}")

        for reporter in reporters:
            reporter_id = reporter.get('id')
            reporter_name = reporter.get('name', reporter_id) # Use name if available

            if not reporter_id:
                logger.warning(f"Warning: Skipping reporter entry with no ID: {reporter}")
                continue

            logger.info(f"\n--- Generating article for reporter: {reporter_name} (ID: {reporter_id}) ---")

            # Construct the command list for subprocess.run
            # Use sys.executable to ensure the correct Python environment is used
            command = [
                sys.executable,
                str(APP_SCRIPT),
                "--auto-generate",
                "--reporter", reporter_id,
                # "--topic", """
                #         This is my origin story, how I came to The Ephergent.
                #         Write an origin story that includes: their ordinary world before transformation,
                #         the inciting incident that changes everything, the struggle/conflict they face,
                #         how they discover or develop their abilities/purpose, and their first step into their new role.
                #         Focus on emotional stakes and character growth""",
                "--skip", skip_steps
            ]

            logger.info(f"Running command: {' '.join(command)}")

            try:
                # Run the command
                # capture_output=True captures stdout/stderr
                # text=True decodes stdout/stderr as text
                # check=True raises CalledProcessError if the command returns a non-zero exit code
                result = subprocess.run(command, capture_output=True, text=True, check=True)
                logger.info(f"Successfully generated article for {reporter_name}.")
                # Optionally log stdout/stderr from the subprocess if needed for debugging
                # logger.debug(f"Subprocess stdout for {reporter_id}:\n{result.stdout}")
                # logger.debug(f"Subprocess stderr for {reporter_id}:\n{result.stderr}")

            except subprocess.CalledProcessError as e:
                logger.error(f"Error generating article for {reporter_name} (ID: {reporter_id}): Command failed with exit code {e.returncode}")
                logger.error(f"Subprocess stderr:\n{e.stderr}")
                logger.error(f"Subprocess stdout:\n{e.stdout}")
            except FileNotFoundError:
                logger.error(f"Error: Could not find the Python executable '{sys.executable}' or the script '{APP_SCRIPT}'.")
                logger.error("Please ensure Python is in your PATH and app.py exists.")
                sys.exit(1)
            except Exception as e:
                logger.error(f"An unexpected error occurred while processing {reporter_name} (ID: {reporter_id}): {e}", exc_info=True)

        logger.info("\n--- Finished generating articles for all reporters ---")

    except json.JSONDecodeError:
        logger.error(f"Error: Could not parse JSON from {PROMPTS_FILE}. Please check the file format.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    generate_articles_for_all_reporters()
