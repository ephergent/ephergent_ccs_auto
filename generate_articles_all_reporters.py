#!/usr/bin/env python3
import json
import subprocess
import sys
from pathlib import Path
import logging
import argparse # Import argparse
from typing import Optional # Import Optional for type hints

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

def generate_articles_for_all_reporters(month: Optional[int], week: Optional[int], generate_denizen: bool, start_index: int):
    """
    Reads reporter IDs from prompts/personality_prompts.json and runs app.py
    to auto-generate an article for each, skipping all steps except generation.
    Optionally specifies month and week for generation, and generates a Denizen.
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

        if not reporters and not generate_denizen:
            logger.info("No reporters found in the prompts file and Denizen generation not requested. Exiting.")
            sys.exit(0)
        elif not reporters and generate_denizen:
            logger.info("No reporters found, but Denizen generation requested. Proceeding with Denizen only.")
        else:
            logger.info(f"Found {len(reporters)} reporters. Starting article generation for each.")


        # Define the steps to skip as per the user's request
        skip_steps = "audio,git,youtube,archive,export,image,video,social"
        logger.info(f"Skipping steps: {skip_steps}")

        current_article_index = start_index

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
            # NEW: Add month and week arguments if provided
            if month is not None:
                command.extend(["--month", str(month)])
            if week is not None:
                command.extend(["--week", str(week)])
            # NEW: Add the current article index
            command.extend(["--index", str(current_article_index)])


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

            current_article_index += 1 # Increment for the next article

        # NEW: Add Denizen generation after all reporters
        if generate_denizen:
            logger.info("\n--- Generating Dimensional Denizen profile ---")
            denizen_command = [
                sys.executable,
                str(APP_SCRIPT),
                "--denizen",
                "--skip", skip_steps # Apply skip steps to denizen generation too
            ]
            # Add month and week arguments if provided
            if month is not None:
                denizen_command.extend(["--month", str(month)])
            if week is not None:
                denizen_command.extend(["--week", str(week)])
            # NEW: Add the current article index for Denizen
            denizen_command.extend(["--index", str(current_article_index)])

            logger.info(f"Running command: {' '.join(denizen_command)}")
            try:
                result = subprocess.run(denizen_command, capture_output=True, text=True, check=True)
                logger.info("Successfully generated Dimensional Denizen profile.")
            except subprocess.CalledProcessError as e:
                logger.error(f"Error generating Denizen profile: Command failed with exit code {e.returncode}")
                logger.error(f"Subprocess stderr:\n{e.stderr}")
                logger.error(f"Subprocess stdout:\n{e.stdout}")
            except Exception as e:
                logger.error(f"An unexpected error occurred while generating Denizen profile: {e}", exc_info=True)

        logger.info("\n--- Finished generating articles for all reporters and Denizen ---")

    except json.JSONDecodeError:
        logger.error(f"Error: Could not parse JSON from {PROMPTS_FILE}. Please check the file format.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate articles for all reporters and optionally a Denizen.")
    parser.add_argument(
        "--month",
        type=int,
        default=None,
        help="Specify the month number for all generated articles (e.g., 1 for Cycle 001)."
    )
    parser.add_argument(
        "--week",
        type=int,
        default=None,
        help="Specify the week number for all generated articles (e.g., 1 for Week 01)."
    )
    parser.add_argument(
        "--denizen",
        action="store_true",
        help="Also generate a Dimensional Denizen profile for the specified date."
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=1, # Default to 1 for the first article
        help="Specify the starting sub-index for generated articles (e.g., 1 for 001_01-01.json)."
    )
    args = parser.parse_args()

    # Basic validation for month/week
    if (args.month is None and args.week is not None) or \
       (args.month is not None and args.week is None):
        parser.error("Both --month and --week must be provided if either is used.")

    generate_articles_for_all_reporters(args.month, args.week, args.denizen, args.start_index)
