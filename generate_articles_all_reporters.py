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
EPH_NEW_STORIES_FILE = Path(__file__).parent / 'input' / 'ephergent_new_stories.json' # NEW CONSTANT

def generate_articles_for_all_reporters(start_month: int, start_week: int, generate_denizen: bool, start_index: int): # UPDATED SIGNATURE
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

    # NEW: Load new story ideas
    if not EPH_NEW_STORIES_FILE.exists():
        logger.error(f"Error: New stories file not found at {EPH_NEW_STORIES_FILE}")
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

        # NEW: Load story ideas
        with open(EPH_NEW_STORIES_FILE, 'r', encoding='utf-8') as f:
            new_stories_data = json.load(f)
        story_ideas = new_stories_data.get('ephergent_story_ideas', [])
        if not story_ideas:
            logger.warning("No story ideas found in ephergent_new_stories.json. Exiting.")
            sys.exit(0)
        logger.info(f"Found {len(story_ideas)} new story ideas to process.")


        # Define the steps to skip as per the user's request
        skip_steps = "audio,git,youtube,archive,export,image,video,social"
        logger.info(f"Skipping steps: {skip_steps}")

        # current_article_index = start_index # REMOVED - now managed per week

        # topic_string = """ """ # REMOVED - now generated per story idea

        current_month = start_month # NEW
        current_week = start_week   # NEW

        for i, story_idea in enumerate(story_ideas): # NEW OUTER LOOP
            # For subsequent stories, increment week and potentially month
            if i > 0: # Only increment for the 2nd story idea onwards
                current_week += 1
                if current_week > 4: # Assuming 4 weeks per month
                    current_week = 1
                    current_month += 1

            logger.info(f"\n--- Processing Story Idea {i+1}: '{story_idea['title']}' for Cycle {current_month:03d}, Week {current_week:02d} ---")

            # Construct the topic string for the LLM
            topic_parts = [
                story_idea['title'],
                story_idea['hook']
            ]
            char_focus_str = []
            # Iterate through character_focus to add details to the topic
            for char_id, focus_desc in story_idea['character_focus'].items():
                # Find reporter name from loaded reporters data for better topic string
                reporter_name = next((r['name'] for r in reporters if r['id'] == char_id), char_id)
                char_focus_str.append(f"{reporter_name}: {focus_desc}")
            if char_focus_str:
                topic_parts.append("Character Focus: " + "; ".join(char_focus_str))

            topic_string = " ".join(topic_parts).strip()
            logger.info(f"Constructed topic: {topic_string[:200]}...") # Log a snippet

            current_article_index_for_week = start_index # Reset index for each new week

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
                    "--topic", topic_string, # Use the dynamically generated topic
                    "--skip", skip_steps
                ]
                # NEW: Add month and week arguments if provided
                # These are now always provided by the outer loop
                command.extend(["--month", str(current_month)]) # UPDATED
                command.extend(["--week", str(current_week)])   # UPDATED
                # NEW: Add the current article index
                command.extend(["--index", str(current_article_index_for_week)]) # UPDATED


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

                current_article_index_for_week += 1 # Increment for the next article in this week

            # NEW: Add Denizen generation after all reporters for the current week
            if generate_denizen:
                logger.info("\n--- Generating Dimensional Denizen profile ---")
                denizen_command = [
                    sys.executable,
                    str(APP_SCRIPT),
                    "--denizen",
                    "--skip", skip_steps # Apply skip steps to denizen generation too
                ]
                # Add month and week arguments if provided
                denizen_command.extend(["--month", str(current_month)]) # UPDATED
                denizen_command.extend(["--week", str(current_week)])   # UPDATED
                # NEW: Add the current article index for Denizen
                denizen_command.extend(["--index", str(current_article_index_for_week)]) # UPDATED

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

            # current_article_index_for_week += 1 # This increment is not needed here, it's reset at the start of the next week's loop

        logger.info("\n--- Finished generating articles for all reporters and Denizen for all story ideas ---") # UPDATED LOG MESSAGE

    except json.JSONDecodeError:
        logger.error(f"Error: Could not parse JSON from {PROMPTS_FILE}. Please check the file format.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate articles for all reporters and optionally a Denizen.")
    # REMOVED --month, --week, --reporter, --topic arguments
    parser.add_argument(
        "--start-month", # UPDATED ARG NAME
        type=int,
        default=1, # UPDATED DEFAULT
        help="Specify the starting month number for the first generated story (e.g., 1 for Cycle 001)." # UPDATED HELP
    )
    parser.add_argument(
        "--start-week", # UPDATED ARG NAME
        type=int,
        default=2, # UPDATED DEFAULT
        help="Specify the starting week number for the first generated story (e.g., 2 for Week 02)." # UPDATED HELP
    )
    parser.add_argument(
        "--denizen",
        action="store_true",
        help="Also generate a Dimensional Denizen profile for the specified date." # UPDATED HELP
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=1, # Default to 1 for the first article
        help="Specify the starting sub-index for generated articles (e.g., 1 for 001_01-01.json)."
    )
    args = parser.parse_args()

    # Basic validation for month/week
    # UPDATED VALIDATION
    if args.start_month is None or args.start_week is None:
        parser.error("Both --start-month and --start-week must be provided.")

    generate_articles_for_all_reporters(args.start_month, args.start_week, args.denizen, args.start_index) # UPDATED CALL
