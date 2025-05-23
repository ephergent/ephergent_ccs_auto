#!/usr/bin/env python3
import json
import logging
import os
from pathlib import Path
import time # Import time for delay

# Import from image_generator.py - Ensure these functions are still valid after refactoring
# We need generate_image_with_comfyui specifically
from utils.image_generator import (
    generate_image_with_comfyui, # Use the main generation function
    DEFAULT_WORKFLOW_FILE,
    # The following are needed by generate_image_with_comfyui but not called directly here
    # open_websocket_connection, queue_prompt, track_progress, get_images_from_history
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('reporter_image_generator.log'), # Log to file
        logging.StreamHandler() # Also log to console
    ]
)
logger = logging.getLogger(__name__)


def load_reporter_prompts(json_path):
    """Load reporter prompts from the JSON file."""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            logger.info(f"Successfully loaded reporter prompts from {json_path}")
            # Return only the list of reporters, seed is not used by generate_image_with_comfyui directly
            return data.get('reporters', [])
    except FileNotFoundError:
        logger.error(f"Reporter prompts file not found: {json_path}")
        return []
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON in reporter prompts file: {json_path}")
        return []
    except Exception as e:
        logger.error(f"Error loading reporter prompts from {json_path}: {e}")
        return []


def generate_reporter_profile_images():
    """Generate profile images for each reporter in the prompts file."""
    # Paths - Use project root directory structure
    project_root = Path(__file__).parent.parent
    json_path = project_root / 'input' / 'reporters_image_prompts.json' # Corrected input path
    # Output directly to assets/images/reporters
    output_dir = project_root / 'assets' / 'images' / 'reporters'
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory set to: {output_dir}")

    # Load reporter prompts
    reporters = load_reporter_prompts(json_path)
    if not reporters:
        logger.error("No reporter prompts loaded. Exiting.")
        return

    logger.info(f"Found {len(reporters)} reporters in the prompts file")

    # Process each reporter
    for idx, reporter_data in enumerate(reporters):
        reporter_id = reporter_data.get('id')
        prompt = reporter_data.get('prompt')

        if not reporter_id or not prompt:
            logger.warning(f"Skipping reporter at index {idx}: missing id or prompt")
            continue

        logger.info(f"--- Generating profile image for reporter: {reporter_id} ---")

        # Create filename (e.g., pixel_paradox.png)
        # Ensure filename matches convention used elsewhere (e.g., video generator)
        filename = f"{reporter_id}.png"
        output_path = output_dir / filename

        # Skip if file already exists
        if output_path.exists():
            logger.info(f"Image already exists for {reporter_id} at {output_path}, skipping.")
            continue

        # Generate image using the standard ComfyUI function
        # It handles workflow loading, modification (prompt, seed), queuing, tracking, and saving
        generated_image_path = generate_image_with_comfyui(
            prompt=prompt,
            workflow_filepath=DEFAULT_WORKFLOW_FILE,
            output_dir=output_dir, # Save directly to the target dir
            filename=filename, # Pass the desired final filename
            change_resolution=True # Ensure this is set to True for resolution changes 1024x1024
        )

        if generated_image_path:
            logger.info(f"Successfully generated profile image for {reporter_id}: {generated_image_path}")
        else:
            logger.error(f"Failed to generate profile image for {reporter_id}")

        # Add a small delay between generations to avoid overwhelming ComfyUI
        time.sleep(5) # Wait 5 seconds before the next image

    logger.info("--- Reporter profile image generation complete ---")


if __name__ == "__main__":
    logger.info("Starting reporter profile image generation script...")
    # Check for ComfyUI URL
    comfy_url = os.getenv('COMFYUI_URL')
    if not comfy_url:
         logger.error("COMFYUI_URL environment variable not set. Cannot connect to ComfyUI.")
    else:
        generate_reporter_profile_images()
    logger.info("Reporter profile image generation script finished.")
