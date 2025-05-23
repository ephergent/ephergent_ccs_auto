#!/usr/bin/env python3
import os
import json # Added for loading JSON
import logging
from pathlib import Path
import re

# --- Configuration ---
PELICAN_CONTENT_DIR = os.getenv('PELICAN_CONTENT_DIR', '../ephergent_blog/content')
if not PELICAN_CONTENT_DIR:
    # Fallback if env var is empty string or None
    PELICAN_CONTENT_DIR = '../ephergent_blog/content'
    print(f"Warning: PELICAN_CONTENT_DIR environment variable not set or empty. Using default: {PELICAN_CONTENT_DIR}")

PELICAN_CONTENT_DIR_PATH = Path(PELICAN_CONTENT_DIR)
PELICAN_PAGES_SUBDIR = "pages"
PELICAN_CHARACTERS_SUBDIR = "characters" # Subdirectory within pages
PROMPTS_JSON_PATH = Path(__file__).parent.parent / 'prompts' / 'personality_prompts.json'
# Point to the directory where character images are expected for the website build
# This assumes the build process copies assets/images/characters to the output/images/characters
CHARACTER_IMAGES_ASSET_DIR = Path(__file__).parent.parent / 'assets' / 'images' / 'characters'
DEFAULT_CHAR_IMAGE_PATH_RELATIVE = '/theme/images/profile.png' # Relative path for web use

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Helper Functions ---
def load_reporter_data(json_path: Path) -> list:
    """Loads reporter data from the specified JSON file."""
    if not json_path.exists():
        logger.error(f"Reporter prompts file not found: {json_path}")
        return []
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        reporters = data.get('reporters', [])
        logger.info(f"Successfully loaded {len(reporters)} reporters from {json_path}")
        return reporters
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON in reporter prompts file: {json_path}")
        return []
    except Exception as e:
        logger.error(f"Error loading reporter prompts from {json_path}: {e}")
        return []

def get_character_image_path(character_slug: str) -> str:
    """Checks if an image exists for the character slug and returns the relative web path."""
    # Use the character_slug which should match the image filename (e.g., pixel_paradox.png)
    expected_image_file = CHARACTER_IMAGES_ASSET_DIR / f"{character_slug}.png"
    if expected_image_file.exists():
        # Return the path relative to the web root, matching pelicanconf.py structure
        return f"/images/characters/{character_slug}.png"
    else:
        logger.warning(f"Image file not found for character '{character_slug}' at {expected_image_file}. Using default.")
        return DEFAULT_CHAR_IMAGE_PATH_RELATIVE

def get_character_role(reporter_data: dict) -> str:
    """Derives a role for the character based on topics or defaults."""
    # Simple logic: Use the first topic if available, otherwise a default.
    topics = reporter_data.get('topics', [])
    if topics:
        # Capitalize the first topic nicely
        role = topics[0].replace('_', ' ').title()
        # Add "Specialist" or "Correspondent" for flavor?
        if "field_report" in topics or "photography" in topics:
            role += " Specialist"
        elif "general" in topics or "news" in topics:
             role += " Correspondent"
        return role
    return "Dimensional Correspondent" # Default role

def get_character_description(reporter_data: dict) -> str:
    """Provides a placeholder description."""
    # TODO: Implement logic to extract description from prompt_file if needed.
    # For now, use a generic description based on name.
    name = reporter_data.get('name', 'This individual')
    return f"{name} reports on events across the multiverse for The Ephergent."

# --- Main Generation Function ---
def generate_character_profile_pages():
    """
    Generates Markdown page files for each character defined in personality_prompts.json.
    These pages will use the 'character_profile.html' template.
    """
    if not PELICAN_CONTENT_DIR_PATH.is_dir():
        logger.error(f"PELICAN_CONTENT_DIR path does not exist or is not a directory: {PELICAN_CONTENT_DIR_PATH}")
        return

    target_dir = PELICAN_CONTENT_DIR_PATH / PELICAN_PAGES_SUBDIR / PELICAN_CHARACTERS_SUBDIR

    try:
        target_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Ensured character pages directory exists: {target_dir}")
    except OSError as e:
        logger.error(f"Error creating character pages directory '{target_dir}': {e}")
        return

    # Load reporter data from JSON
    reporters = load_reporter_data(PROMPTS_JSON_PATH)
    if not reporters:
        logger.error("No reporter data loaded. Cannot generate character pages.")
        return

    generated_count = 0
    skipped_count = 0
    error_count = 0

    # Add Ephergent One manually if not in JSON (or handle separately)
    # For simplicity, let's assume it might be added to JSON later or handled by theme directly.
    # We will only generate pages for reporters listed in the JSON for now.

    for reporter_data in reporters:
        name = reporter_data.get('name')
        reporter_id = reporter_data.get('id') # Use 'id' for slug and image lookup

        if not name or not reporter_id:
            logger.warning(f"Skipping reporter due to missing 'name' or 'id': {reporter_data}")
            skipped_count += 1
            continue

        # Use the reporter ID (which should be filename-safe) as the slug
        slug = reporter_id
        # Pass the slug to get the correct image path format
        image_path = get_character_image_path(slug)
        role = get_character_role(reporter_data)
        description = get_character_description(reporter_data) # Placeholder

        # Define the markdown file path
        md_filename = f"{slug}.md"
        target_md_path = target_dir / md_filename

        # Create the simple Markdown content for the page
        # It mainly needs the Title, Slug, and Template metadata.
        # The character_profile.html template should be updated to pull details
        # from a global context variable populated from personality_prompts.json
        # OR, we can embed the necessary details here for the template to use directly.
        # Let's embed them for now, assuming the template can read them.
        markdown_content = f"""Title: {name}
Slug: {slug}
Template: character_profile
Status: published
Role: {role}
Image: {image_path}
Description: {description}

<!--
This content is minimal. The 'character_profile.html' template should use the
metadata fields (Title, Slug, Role, Image, Description) defined above.
Alternatively, the template could load data directly from a site-wide context
variable populated from personality_prompts.json during the Pelican build.
-->
"""

        try:
            with open(target_md_path, "w", encoding="utf-8") as f:
                f.write(markdown_content.strip() + "\n") # Ensure clean output
            logger.info(f"Generated profile page for '{name}' ({slug}) at: {target_md_path}")
            generated_count += 1
        except IOError as e:
            logger.error(f"Failed to write profile page for '{name}' to {target_md_path}: {e}")
            error_count += 1

    logger.info(f"Character page generation complete. Generated: {generated_count}, Skipped: {skipped_count}, Errors: {error_count}")


if __name__ == "__main__":
    logger.info("Starting character profile page generation...")
    generate_character_profile_pages()
    logger.info("Character profile page generation finished.")
    print("\nRun your Pelican build command again to include the new character pages.")
    # Construct the example command path more robustly
    pelican_base_dir = PELICAN_CONTENT_DIR_PATH.parent
    print(f"Example: cd \"{pelican_base_dir}\" && pelican content -s pelicanconf.py -o output -t themes/ephergent-theme")
