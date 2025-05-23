#!/usr/bin/env python3
import os
import logging
import argparse
import random
import shutil
import sys
import json
import re
from pathlib import Path
from datetime import datetime, timedelta
from dotenv import load_dotenv
from typing import Dict, Any, Optional, List # Typing imports

# --- Utility Imports ---
from utils.reporter import Reporter
from utils.topic_generator import generate_topic
from utils.article import generate_article
from utils.summarize import generate_summary
from utils.title import generate_titles
from utils.image_generator import generate_ephergent_image, generate_article_essence_image_prompt # Import single prompt function
from utils.audio_generator import generate_article_audio, prepare_denizen_text_for_tts
from utils.pelican_exporter import (
    format_story_markdown, format_denizen_pelican_markdown,
    export_to_pelican, sanitize_filename, PELICAN_ARTICLES_SUBDIR,
    PELICAN_IMAGES_SUBDIR, PELICAN_AUDIO_SUBDIR, PELICAN_CONTENT_DIR as PELICAN_CONTENT_DIR_PATH, # Import path for direct use
    PELICAN_DENIZENS_SUBDIR
)
from utils.video_generator import generate_youtube_video
from utils.youtube_uploader import upload_to_youtube
from utils.git_publisher import publish_to_git
from utils.social_publisher import post_article_to_social_media
from utils.mailgun_sender import send_email
from utils.archiver import Archiver, BASE_ARCHIVE_DIR # Import Archiver class and BASE_ARCHIVE_DIR
from utils.metadata_utils import extract_pelican_metadata # For parsing archived markdown
# Import the Denizen profile generator (needed for checking if denizen components are available)
try:
    from utils.profile_image_generator import generate_denizen_profile
except ImportError as e:
    # Logger might not be ready yet, print instead
    print(f"Warning: Failed to import required Denizen components: {e}. Denizen workflow will not be available.")
    generate_denizen_profile = None
    # prepare_denizen_text_for_tts is imported from audio_generator

# --- Logger Setup (Initialize early) ---
# Load environment variables early in case logger config depends on them
load_dotenv()
LOG_FILE_PATH = Path(__file__).parent / 'ephergent_content_creator.log'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE_PATH), # Log to file
        logging.StreamHandler() # Also log to console
    ]
    # force=True # Use force=True in Python 3.8+ to reconfigure logging if needed
)
# Check if handlers are already configured (e.g., if run multiple times in same process)
if not logging.getLogger().handlers:
     logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(LOG_FILE_PATH), # Log to file
            logging.StreamHandler() # Also log to console
        ]
    )

logger = logging.getLogger(__name__)
logger.info("Logger initialized.") # Confirm initialization

# --- Configuration ---
OUTPUT_DIR = Path(__file__).parent / 'output'
OUTPUT_DIR.mkdir(exist_ok=True)
# INPUT_STORIES_FILE is deprecated
# Use 'ready' subdirectory for stories to be processed
INPUT_STORIES_DIR_READY = Path(__file__).parent / 'input' / 'stories' / 'ready'
INPUT_STORIES_DIR_READY.mkdir(parents=True, exist_ok=True) # Ensure 'ready' dir exists
ARCHIVED_STORIES_DIR = Path(__file__).parent / 'input' / 'stories' / 'archived'
ARCHIVED_STORIES_DIR.mkdir(parents=True, exist_ok=True) # Ensure 'archived' dir exists

BLOG_URL = os.getenv('BLOG_URL', 'https://ephergent.com') # Default blog URL
CLEANUP_ARTIFACTS = os.getenv('CLEANUP_ARTIFACTS', 'true').lower() in ('true', '1', 'yes', 't')
SKIP_STEPS = os.getenv('SKIP_STEPS', '').lower().split(',') # e.g., SKIP_STEPS=audio,social,mail,git,youtube,archive,export,image,video
CLIENT_SECRETS_FILE = os.getenv('CLIENT_SECRETS_FILE', 'secrets/client_secret.json')
SEASON_START_DATE = datetime(2025, 6, 1)
# NUM_ARTICLE_IMAGES environment variable is removed as per TODO

# Define valid steps for regeneration
VALID_REGEN_STEPS = {'image', 'feature_image', 'article_image', 'audio', 'video', 'youtube', 'mail', 'social'}

# --- Reporter Mapping ---
# Simple mapping from "Filed by" string to reporter ID
# This might need expansion or loading from config if more complex names are used
REPORTER_NAME_TO_ID_MAP = {
    "Pixel Paradox, Interdimensional Correspondent": "pixel_paradox",
    "Pixel Paradox": "pixel_paradox",
    "Vex Parallax": "vex_parallax", # Added Vex Parallax
    # Add other reporters here if they appear in the JSON
}

# --- Ephergent Universe Dimensions/Themes ---
# Full list of themes/concepts (used for general context/prompts)
dimension_themes = [
    "Urban Sci-Fi Prime Material", "Gothic Horror Nocturne", "Steampunk Cogsworth",
    "Cyberpunk AI Mechanica", "Ecological Sci-Fi Verdantia", "Cosmic Horror The Edge",
    "Time-Travel Mystery Chronos Reach", "Absurdist Bureaucracy", "Political Thriller",
    "Reality Stabilization", "Narrative Causality", "Interdimensional Economics", "Market Volatility",
    "Data-Driven Divination", "Temporal Anomaly Repair", "Memetic Warfare"
]

# Core Dimensions (used for primary location tag)
CORE_DIMENSIONS = [
    "Prime Material", "Nocturne Aeturnus", "Cogsworth Cogitarium", "Verdantia", "The Edge"
]


# --- Helper Functions ---
def get_reporter_from_name(filed_by: str) -> Reporter | None:
    """Gets the Reporter object based on the 'Filed by' string."""
    # Attempt direct match first
    reporter_id = REPORTER_NAME_TO_ID_MAP.get(filed_by)
    if reporter_id:
        logger.info(f"Mapped '{filed_by}' to Reporter ID: '{reporter_id}' via direct map.")
    else:
        # If no direct map, try finding reporter by name/ID using Reporter class method
        try:
            reporter_obj = Reporter(identifier=filed_by)
            if reporter_obj and reporter_obj.reporter_data:
                reporter_id = reporter_obj.id
                logger.info(f"Found reporter for '{filed_by}' using Reporter class lookup. ID: '{reporter_id}'. Using this ID.")
            else:
                logger.warning(f"No reporter found for name/ID: '{filed_by}'. Using default 'pixel_paradox'.")
                reporter_id = "pixel_paradox" # Example fallback
        except Exception as e:
             logger.warning(f"Error looking up reporter '{filed_by}': {e}. Using default 'pixel_paradox'.")
             reporter_id = "pixel_paradox"

    try:
        reporter = Reporter(identifier=reporter_id)
        if not reporter.reporter_data:
            logger.error(f"Could not load data for reporter ID '{reporter_id}' mapped from '{filed_by}'.")
            return None
        logger.info(f"Using Reporter: ID='{reporter.id}', Name='{reporter.name}'")
        return reporter
    except Exception as e:
        logger.error(f"Error initializing reporter '{reporter_id}' from name '{filed_by}': {e}", exc_info=True)
        return None


def calculate_publish_date(month: int, week: int, start_date: datetime) -> datetime:
    """Calculates the target publish date based on month, week, and season start."""
    # Assuming 4 weeks per month for simplicity, adjust if needed
    # Weeks are 1-indexed, Months are 1-indexed
    # Week 1 of Month 1 is day 0-6 offset
    # Week 1 of Month 2 is day 28-34 offset
    # Week W of Month M is (M-1)*28 + (W-1)*7 days offset
    days_offset = (month - 1) * 28 + (week - 1) * 7
    return start_date + timedelta(days=days_offset)

def cleanup_output_directory(output_subdir: Path):
    """Removes the temporary output subdirectory if cleanup is enabled."""
    if not CLEANUP_ARTIFACTS:
        logger.info("Skipping cleanup of temporary output directory (CLEANUP_ARTIFACTS is not 'true').")
        return
    if output_subdir and output_subdir.exists() and output_subdir.is_dir():
        try:
            shutil.rmtree(output_subdir)
            logger.info(f"Successfully cleaned up temporary output directory: {output_subdir}")
        except Exception as e:
            logger.error(f"Error cleaning up temporary output directory {output_subdir}: {e}", exc_info=True)
    else:
        logger.warning(f"Temporary output directory not found or is not a directory: {output_subdir}")

def get_next_story_id() -> tuple[int, int]:
    """
    Scans the archived stories directory AND the ready stories directory
    to find the highest MMM_WW.json file and calculates the next sequential
    month and week.
    """
    logger.info(f"Determining next story ID from directories: {ARCHIVED_STORIES_DIR} and {INPUT_STORIES_DIR_READY}")
    max_month = 0
    max_week = 0
    pattern = re.compile(r"(\d{3})_(\d{0,2})\.json") # Allow 0 or 2 digits for week

    # Combine files from both directories
    all_story_files = []
    if ARCHIVED_STORIES_DIR.is_dir():
        all_story_files.extend(ARCHIVED_STORIES_DIR.glob("*.json"))
    else:
        logger.warning(f"Archived stories directory not found: {ARCHIVED_STORIES_DIR}.")

    if INPUT_STORIES_DIR_READY.is_dir():
        all_story_files.extend(INPUT_STORIES_DIR_READY.glob("*.json"))
    else:
        logger.warning(f"Ready stories directory not found: {INPUT_STORIES_DIR_READY}.")


    if not all_story_files:
        logger.info("No MMM_WW.json files found in either directory. Starting with 001_01.")
        return (1, 1)

    for file_path in all_story_files:
        match = pattern.match(file_path.name)
        if match:
            month = int(match.group(1))
            # Handle cases where week might be missing or malformed, default to 0
            try:
                week = int(match.group(2)) if match.group(2) else 0
            except ValueError:
                week = 0
                logger.warning(f"Could not parse week from filename {file_path.name}. Treating as week 0.")

            if month > max_month:
                max_month = month
                max_week = week # When a new max month is found, its week becomes the max week
            elif month == max_month and week > max_week:
                max_week = week

    if max_month == 0:
         # This case should ideally be caught by the initial check for all_story_files,
         # but as a safeguard, if we somehow processed files but found no valid pattern matches.
         logger.warning("No valid MMM_WW.json patterns found in files. Starting with 001_01.")
         return (1, 1)


    # Calculate next ID
    next_month = max_month
    next_week = max_week + 1

    # Assuming 4 weeks per month for this numbering scheme
    if next_week > 4:
        next_week = 1
        next_month += 1

    logger.info(f"Highest ID found across ready and archived: {max_month:03d}_{max_week:02d}. Next ID will be: {next_month:03d}_{next_week:02d}.")
    return (next_month, next_week)


# --- Regeneration Workflow ---
def regenerate_artifacts(archive_id: str, steps_to_regenerate: List[str]):
    """
    Regenerates specific artifacts for a previously archived item.

    Args:
        archive_id (str): The name of the archive directory (e.g., 'YYYY-MM-DD-slug').
        steps_to_regenerate (List[str]): List of steps to perform ('image', 'audio', etc.).
    """
    logger.info(f"--- Starting Regeneration for Archive ID: {archive_id} ---")
    logger.info(f"Steps to regenerate: {', '.join(steps_to_regenerate)}")

    target_archive_dir = BASE_ARCHIVE_DIR / archive_id
    if not target_archive_dir.is_dir():
        logger.error(f"Archive directory not found: {target_archive_dir}")
        return

    # --- Find and Load Original Data ---
    markdown_files = list(target_archive_dir.glob("*.md"))
    if not markdown_files:
        logger.error(f"No markdown file found in archive: {target_archive_dir}")
        return
    archived_markdown_path = markdown_files[0]
    if len(markdown_files) > 1:
        logger.warning(f"Multiple markdown files found in {target_archive_dir}. Using: {archived_markdown_path.name}")

    logger.info(f"Loading data from archived markdown: {archived_markdown_path.name}")
    markdown_content = ""
    content_body = ""
    metadata = {}
    try:
        with open(archived_markdown_path, 'r', encoding='utf-8') as f:
            markdown_content = f.read()
        metadata, content_body = extract_pelican_metadata(markdown_content) # Use extract_pelican_metadata
        metadata['content'] = content_body # Add content body to metadata dict for consistency
    except Exception as e:
        logger.error(f"Failed to read or parse markdown file {archived_markdown_path}: {e}", exc_info=True)
        return

    # Extract necessary info (handle potential missing keys)
    title = metadata.get('title', archive_id) # Fallback title
    author_name = metadata.get('author', 'Pixel Paradox') # Fallback author
    category = metadata.get('category', 'Unknown')
    tags_str = metadata.get('tags', '[]')
    try: tags = json.loads(tags_str) if isinstance(tags_str, str) else (tags_str if isinstance(tags_str, list) else [])
    except json.JSONDecodeError: tags = []
    original_youtube_url = metadata.get('youtube_url') # Get original URL if present
    is_denizen = category.lower() == "dimensional denizen"
    location = metadata.get('location', 'Unknown Location') # Get location from metadata

    logger.info(f"Loaded Metadata: Title='{title}', Author='{author_name}', Category='{category}', Denizen={is_denizen}, Location='{location}'")

    # --- Get Reporter ---
    reporter = get_reporter_from_name(author_name)
    if not reporter:
        logger.error(f"Could not initialize reporter '{author_name}'. Cannot proceed.")
        return

    # --- Setup Temp Dir ---
    regen_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    temp_regen_dir = OUTPUT_DIR / f"regen_{archive_id}_{regen_timestamp}"
    temp_regen_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Using temporary directory for regeneration: {temp_regen_dir}")

    # --- Locate Existing Artifacts in Archive ---
    archive_images_dir = target_archive_dir / "images"
    archive_audio_dir = target_archive_dir / "audio"
    archive_video_dir = target_archive_dir / "video"

    # Find existing files (handle potential non-existence)
    existing_feature_image = next(archive_images_dir.glob(f"{archive_id}_feature.*"), None)
    # Article images are now expected to be a single file named like *_article_essence.*
    existing_article_image = next(archive_images_dir.glob(f"{archive_id}_article_essence.*"), None)
    existing_audio = next(archive_audio_dir.glob(f"{archive_id}_audio*.*"), None) # Allow _combined suffix
    existing_video = next(archive_video_dir.glob(f"{archive_id}_video.*"), None)
    existing_prompts_file = next(target_archive_dir.glob(f"{archive_id}_image_prompts.json"), None)

    logger.info(f"Found existing artifacts: FeatureImg={existing_feature_image}, ArticleImg={existing_article_image}, Audio={existing_audio}, Video={existing_video}")

    # --- Execute Regeneration Steps ---
    newly_generated_artifacts = {} # Store paths of *newly* generated files in temp dir
    image_prompt_details = [] # Store new prompts if images are regenerated
    if existing_prompts_file and 'image' not in steps_to_regenerate and 'feature_image' not in steps_to_regenerate and 'article_image' not in steps_to_regenerate:
        # Load existing prompts if not regenerating images
        try:
            with open(existing_prompts_file, 'r', encoding='utf-8') as f:
                image_prompt_details = json.load(f)
            logger.info(f"Loaded existing image prompts from {existing_prompts_file.name}")
        except Exception as e:
            logger.warning(f"Could not load existing image prompts file: {e}")
            image_prompt_details = []

    # Prepare story_data dict for image generation (using loaded content and metadata)
    story_data_for_images = {
        'title': title,
        'content': content_body, # Use the extracted content body
        'location': location, # Use location from metadata
        'stardate': metadata.get('stardate', 'N/A'), # Get stardate from metadata
        'featured_characters': metadata.get('featured_characters', []), # Get featured_characters from metadata
    }


    # Regenerate Images
    regen_feature = 'image' in steps_to_regenerate or 'feature_image' in steps_to_regenerate
    regen_article = 'image' in steps_to_regenerate or 'article_image' in steps_to_regenerate

    if regen_feature:
        logger.info("Regenerating Feature Image...")
        temp_img_dir = temp_regen_dir / "images"
        temp_img_dir.mkdir(exist_ok=True)
        feature_image_filename = f"{archive_id}_feature.png" # Use archive_id as base
        # Remove old feature prompt before adding new one
        image_prompt_details = [p for p in image_prompt_details if p.get("image_type") != "featured"]
        new_feature_path = generate_ephergent_image(
            story_data=story_data_for_images, reporter=reporter, image_type="featured",
            output_dir=temp_img_dir, filename=feature_image_filename,
            prompt_details_list=image_prompt_details # Pass list to store prompt
        )
        if new_feature_path: newly_generated_artifacts["feature_image"] = new_feature_path
        else: logger.warning("Failed to regenerate feature image.")

    if regen_article:
        logger.info(f"Regenerating single Article Essence Image...")
        temp_img_dir = temp_regen_dir / "images"
        temp_img_dir.mkdir(exist_ok=True)
        new_article_path = None # Will hold the single path

        # Remove old article prompts before adding new ones
        image_prompt_details = [p for p in image_prompt_details if p.get("image_type") != "article"]

        # Generate the single prompt
        article_essence_prompt = generate_article_essence_image_prompt(
            story_data=story_data_for_images, # Pass the dict with full content and location
            reporter=reporter
        )

        if not article_essence_prompt:
            logger.error("Failed to generate article essence prompt for regeneration.")
        else:
            # Store the newly generated prompt first
            article_image_filename = f"{archive_id}_article_essence.png" # Use consistent name
            prompt_info = {
                "image_type": "article",
                "filename": article_image_filename,
                "prompt": article_essence_prompt,
                "index": 0 # Index 0 for the single image
            }
            # Avoid duplicates if somehow called multiple times
            if not any(p.get("filename") == article_image_filename for p in image_prompt_details):
                image_prompt_details.append(prompt_info)
                logger.info(f"Stored regenerated prompt details for {article_image_filename}")

            # Generate image using the pre-generated prompt
            logger.info(f"Generating Article Essence Image using pre-generated prompt...")
            new_article_path = generate_ephergent_image(
                story_data=story_data_for_images, reporter=reporter, image_type="article",
                output_dir=temp_img_dir, filename=article_image_filename, image_index=0,
                prompt_override=article_essence_prompt, # Use the override
                prompt_details_list=image_prompt_details # Pass list (already populated)
            )
            if new_article_path: newly_generated_artifacts["article_images"] = new_article_path # Store single path
            else: logger.warning(f"Failed to regenerate article essence image.")

    # Save updated image prompts JSON to temp dir if images were regenerated
    if (regen_feature or regen_article) and image_prompt_details:
        temp_prompts_path = temp_regen_dir / f"{archive_id}_image_prompts.json"
        try:
            # Ensure uniqueness before saving
            seen_filenames = set()
            unique_prompts = []
            for p in image_prompt_details:
                fname = p.get("filename")
                if fname not in seen_filenames:
                    unique_prompts.append(p)
                    seen_filenames.add(fname)

            with open(temp_prompts_path, 'w', encoding='utf-8') as f:
                json.dump(unique_prompts, f, indent=2)
            newly_generated_artifacts["image_prompts_file"] = temp_prompts_path
            logger.info(f"Saved regenerated image prompts details to temporary file: {temp_prompts_path.name}")
        except Exception as e:
            logger.error(f"Failed to save regenerated image prompts JSON: {e}")


    # Regenerate Audio
    if 'audio' in steps_to_regenerate:
        logger.info("Regenerating Audio...")
        temp_audio_dir = temp_regen_dir / "audio" # Save to temp audio subdir
        temp_audio_dir.mkdir(exist_ok=True)
        new_audio_path = None
        if is_denizen:
            # Need to reconstruct denizen data for TTS prep
            logger.warning("Denizen audio regeneration requires specific data extraction - using basic content.")
            denizen_data_for_tts = {'name': title.replace("Daily Dimensional Denizen: ",""), 'details': {}} # Basic reconstruction
            denizen_audio_text = prepare_denizen_text_for_tts(denizen_data_for_tts, content_body) # Use content_body
            if denizen_audio_text:
                 new_audio_path = generate_article_audio(
                     reporter=reporter, article_content=denizen_audio_text, title=title,
                     output_dir=temp_regen_dir, filename_base=archive_id, speed=1.1 # Save to base temp dir
                 )
        else:
            new_audio_path = generate_article_audio(
                reporter=reporter, article_content=content_body, title=title, # Use content_body
                output_dir=temp_regen_dir, filename_base=archive_id, speed=1.1 # Save to base temp dir
            )

        if new_audio_path: newly_generated_artifacts["audio"] = new_audio_path
        else: logger.warning("Failed to regenerate audio.")

    # Regenerate Video
    if 'video' in steps_to_regenerate:
        logger.info("Regenerating Video...")
        # Determine inputs: use newly generated if available, else use archived
        audio_input_path = newly_generated_artifacts.get("audio", existing_audio)
        feature_img_input_path = newly_generated_artifacts.get("feature_image", existing_feature_image)

        # Determine the list of article images for the video (will be 0 or 1 for standard articles)
        article_imgs_for_regen_video = []
        if newly_generated_artifacts.get("article_images"):
             article_imgs_for_regen_video = [newly_generated_artifacts["article_images"]]
        elif existing_article_image: # Use the single existing article image if no new one was generated
             article_imgs_for_regen_video = [existing_article_image]
             logger.info(f"Using existing article image for regenerated video.")
        # Note: If it's a Denizen and not regenerating images, existing_article_images (plural) might be needed.
        # The current logic assumes standard article regeneration. Denizen regeneration might need refinement.
        # For now, assuming standard article regen where only one article image is expected.


        if not audio_input_path: logger.error("Cannot regenerate video: Audio path is missing.")
        elif not feature_img_input_path: logger.error("Cannot regenerate video: Feature image path is missing.")
        else:
            temp_video_dir = temp_regen_dir / "video"
            temp_video_dir.mkdir(exist_ok=True)
            new_video_path = generate_youtube_video(
                reporter=reporter, title=title, audio_path=audio_input_path,
                featured_image_path=feature_img_input_path,
                article_image_paths=article_imgs_for_regen_video, # Use the determined list (0 or 1 item)
                output_dir=temp_video_dir, filename_base=archive_id
            )
            if new_video_path: newly_generated_artifacts["video"] = new_video_path
            else: logger.warning("Failed to regenerate video.")

    # Regenerate YouTube Upload
    new_youtube_url = original_youtube_url # Start with original
    if 'youtube' in steps_to_regenerate:
        logger.info("Regenerating YouTube Upload...")
        video_input_path = newly_generated_artifacts.get("video", existing_video)
        thumb_input_path = newly_generated_artifacts.get("feature_image", existing_feature_image)

        if not video_input_path: logger.error("Cannot regenerate YouTube upload: Video path is missing.")
        else:
            # Regenerate description/summary (optional, could reuse original)
            first_paragraph = content_body.split('\n')[0] if content_body else "No content available."
            video_description = ' '.join(first_paragraph.split()[:75]) + "...\n\nRead the full report: [Link Placeholder]"
            video_tags_regen = tags + ["The Ephergent", reporter.name.replace(" ", "")]

            new_youtube_id = upload_to_youtube(
                video_file_path=video_input_path, title=title, description=video_description,
                tags=video_tags_regen, thumbnail_path=thumb_input_path
            )
            if new_youtube_id:
                new_youtube_url = f"https://www.youtube.com/watch?v={new_youtube_id}"
                logger.info(f"YouTube re-upload successful: {new_youtube_url}")
                # Update description placeholder (for potential markdown update)
                # video_description = video_description.replace("[Link Placeholder]", f"{BLOG_URL}/{PELICAN_ARTICLES_SUBDIR}/{archive_id}.html")
            else:
                logger.warning("YouTube re-upload failed.")
                new_youtube_url = original_youtube_url # Revert to original on failure

    # --- Update Archive and Pelican Content ---
    logger.info("Updating archive and Pelican content with regenerated artifacts...")
    pelican_updated = False
    commit_message_details = []

    # Helper to copy and log
    def copy_and_log(src: Path, target_dir: Path, artifact_type: str) -> bool:
        # nonlocal pelican_updated # Allow modification of outer scope variable - not needed if only returning bool
        if src and src.exists():
            dest_path = target_dir / src.name
            try:
                target_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dest_path)
                logger.info(f"Updated {artifact_type} in {target_dir.name}: {dest_path.name}")
                return True
            except Exception as e:
                logger.error(f"Failed to copy {artifact_type} {src.name} to {target_dir}: {e}")
                return False
        return False # Return False if source is None or doesn't exist

    # Update Archive
    if newly_generated_artifacts.get("feature_image"):
        if copy_and_log(newly_generated_artifacts["feature_image"], archive_images_dir, "archive feature image"):
             commit_message_details.append("feature image")
             pelican_updated = True # Assume archive update implies Pelican update needed
    # Handle single article image update
    if newly_generated_artifacts.get("article_images"):
        new_article_path_single = newly_generated_artifacts["article_images"]
        if copy_and_log(new_article_path_single, archive_images_dir, "archive article image"):
            commit_message_details.append("article image") # Singular
            pelican_updated = True # Assume archive update implies Pelican update needed
    if newly_generated_artifacts.get("audio"):
        if copy_and_log(newly_generated_artifacts["audio"], archive_audio_dir, "archive audio"):
            commit_message_details.append("audio")
            pelican_updated = True # Assume archive update implies Pelican update needed
    if newly_generated_artifacts.get("video"):
        if copy_and_log(newly_generated_artifacts["video"], archive_video_dir, "archive video"):
            commit_message_details.append("video")
            # Video doesn't go into Pelican content dir, but might trigger markdown update (YouTube URL)
    if newly_generated_artifacts.get("image_prompts_file"):
        copy_and_log(newly_generated_artifacts["image_prompts_file"], target_archive_dir, "archive prompts JSON")
        # Don't necessarily add prompts file to commit message

    # Update Pelican Content Directory (if path exists)
    if PELICAN_CONTENT_DIR_PATH and Path(PELICAN_CONTENT_DIR_PATH).is_dir():
        pelican_img_dir = Path(PELICAN_CONTENT_DIR_PATH) / PELICAN_IMAGES_SUBDIR
        pelican_audio_dir = Path(PELICAN_CONTENT_DIR_PATH) / PELICAN_AUDIO_SUBDIR
        # Determine correct subdir for markdown (articles or denizens)
        pelican_md_subdir_rel = PELICAN_ARTICLES_SUBDIR if not is_denizen else PELICAN_DENIZENS_SUBDIR
        pelican_md_dir = Path(PELICAN_CONTENT_DIR_PATH) / pelican_md_subdir_rel
        pelican_article_path = pelican_md_dir / f"{archive_id}.md"

        # Copy regenerated media to Pelican content dir
        if newly_generated_artifacts.get("feature_image"):
            if copy_and_log(newly_generated_artifacts["feature_image"], pelican_img_dir, "Pelican feature image"):
                pelican_updated = True
        # Handle single article image update for Pelican
        if newly_generated_artifacts.get("article_images"):
            new_article_path_single = newly_generated_artifacts["article_images"]
            if copy_and_log(new_article_path_single, pelican_img_dir, "Pelican article image"):
                pelican_updated = True
        if newly_generated_artifacts.get("audio"):
            if copy_and_log(newly_generated_artifacts["audio"], pelican_audio_dir, "Pelican audio"):
                pelican_updated = True

        # Update YouTube URL in Pelican Markdown metadata AND content if it changed or needs adding
        if new_youtube_url != original_youtube_url or (new_youtube_url and not original_youtube_url):
            logger.info(f"YouTube URL changed or added. Updating Pelican markdown: {pelican_article_path.name}")
            if pelican_article_path.exists():
                try:
                    with open(pelican_article_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                    new_lines = []
                    metadata_updated = False
                    content_html_updated = False
                    metadata_end_index = -1 # Index where metadata ends (first blank line after metadata)
                    content_start_index = -1 # Index where content body starts

                    # Find existing URL line, the end of metadata, and the start of content
                    for i, line in enumerate(lines):
                        # Update metadata line
                        if line.strip().lower().startswith('youtube_url:'):
                            new_lines.append(f"YouTube_Url: {new_youtube_url}\n") # Keep consistent capitalization
                            metadata_updated = True
                        elif line.strip().lower().startswith('youtubeurl:'): # Handle case from format_story_markdown
                            new_lines.append(f"YouTubeURL: {new_youtube_url}\n")
                            metadata_updated = True
                        else:
                            new_lines.append(line)

                        # Check for the end of the metadata block (first blank line)
                        if metadata_end_index == -1 and line.strip() == "" and i > 0 and lines[i-1].strip() != "":
                             metadata_end_index = i
                             content_start_index = i + 1 # Content starts after the blank line

                    # If the metadata URL line was not found, add it at the end of the metadata block
                    if not metadata_updated and new_youtube_url:
                        if metadata_end_index != -1:
                            # Insert the new line before the content body starts
                            new_lines.insert(metadata_end_index, f"YouTubeURL: {new_youtube_url}\n")
                            logger.info("Added new YouTubeURL metadata line to Pelican markdown.")
                            metadata_updated = True
                            # Adjust content_start_index if we inserted before it
                            if content_start_index != -1:
                                content_start_index += 1
                        else:
                            # Fallback: Add it after the last metadata line if no blank line exists
                            last_metadata_line = -1
                            for i in range(len(lines)):
                                if re.match(r'^[A-Za-z0-9_-]+:\s*.*$', lines[i].strip()):
                                    last_metadata_line = i
                                elif lines[i].strip() == "":
                                     break # Exit loop, metadata_end_index should have been set
                                else:
                                     break # Exit loop

                            if last_metadata_line != -1:
                                new_lines.insert(last_metadata_line + 1, f"YouTubeURL: {new_youtube_url}\n")
                                logger.info("Added new YouTubeURL metadata line after last metadata line.")
                                metadata_updated = True
                                # Adjust content_start_index if we inserted before it
                                if content_start_index != -1 and last_metadata_line + 1 <= content_start_index:
                                     content_start_index += 1
                            else:
                                logger.warning("Could not find a clear place to insert YouTubeURL metadata in Pelican markdown.")


                    # --- Update/Add YouTube HTML Embed in Content ---
                    if new_youtube_url:
                        # Extract video ID
                        match_id = re.search(r"v=([^&]+)", new_youtube_url)
                        video_id = match_id.group(1) if match_id else None

                        if video_id:
                            # Construct the new HTML embed block
                            new_embed_html_lines = [
                                '<figure>\n',
                                '  <div style="position: relative; padding-bottom: 56.25%; height: 0; overflow: hidden; max-width: 100%; height: auto;">\n',
                                f'    <iframe src="https://www.youtube.com/embed/{video_id}" style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>\n',
                                '  </div>\n',
                                '  <figcaption style="font-size: 0.8em; color: grey;">⁂ Video created by The Ephergent\'s dimensionally-aware AI ⁂</figcaption>\n',
                                '</figure>\n'
                            ]

                            # Search for existing iframe and replace
                            iframe_start_index = -1
                            iframe_end_index = -1
                            # Search only within the content body lines if content_start_index is known
                            search_lines = new_lines[content_start_index:] if content_start_index != -1 else new_lines

                            for i, line in enumerate(search_lines):
                                # Adjust index back to original list
                                original_index = i + (content_start_index if content_start_index != -1 else 0)
                                if '<iframe' in line and 'youtube.com/embed' in line:
                                    iframe_start_index = original_index
                                    # Find the end of the figure/iframe block
                                    for j in range(original_index, len(new_lines)):
                                        if '</figure>' in new_lines[j] or '</iframe>' in new_lines[j]:
                                            iframe_end_index = j
                                            break
                                    break # Found the existing iframe block

                            if iframe_start_index != -1 and iframe_end_index != -1:
                                # Replace the old block with the new one
                                logger.info(f"Found existing YouTube embed HTML (lines {iframe_start_index}-{iframe_end_index}), replacing.")
                                new_lines[iframe_start_index : iframe_end_index + 1] = new_embed_html_lines
                                content_html_updated = True
                            elif content_start_index != -1:
                                # No existing iframe found, insert the new block
                                # Find insertion point: before '---' or at the end of content body
                                insertion_index = len(new_lines) # Default to end
                                for i in range(content_start_index, len(new_lines)):
                                    if new_lines[i].strip() == '---':
                                        insertion_index = i
                                        break # Insert before the horizontal rule

                                logger.info(f"No existing YouTube embed HTML found, inserting at line {insertion_index}.")
                                # Insert the new lines
                                new_lines[insertion_index:insertion_index] = new_embed_html_lines
                                content_html_updated = True
                            else:
                                logger.warning("Could not determine content start index to insert YouTube embed HTML.")


                    # Write the updated lines back to the file
                    if metadata_updated or content_html_updated:
                        with open(pelican_article_path, 'w', encoding='utf-8') as f:
                            f.writelines(new_lines)
                        logger.info("Successfully updated Pelican markdown file.")
                        pelican_updated = True
                        # Add details to commit message
                        if metadata_updated and content_html_updated:
                             commit_message_details.append("YouTube URL and embed")
                        elif metadata_updated:
                             commit_message_details.append("YouTube URL metadata")
                        elif content_html_updated:
                             commit_message_details.append("YouTube embed")
                    else:
                        logger.warning("Pelican markdown file was not updated (no changes to YouTube URL or embed).")

                except Exception as e:
                    logger.error(f"Failed to update YouTube URL or embed in {pelican_article_path}: {e}", exc_info=True)
            else:
                logger.warning(f"Pelican markdown file not found, cannot update YouTube URL or embed: {pelican_article_path}")

    else:
        logger.warning("PELICAN_CONTENT_DIR not configured or found. Skipping update of Pelican files.")

    # --- Commit Changes to Git ---
    # Only commit if Pelican content directory was updated (either files copied or markdown changed)
    if pelican_updated:
        # Ensure commit message details are unique and sorted for consistency
        unique_details = sorted(list(set(commit_message_details)))
        if not unique_details: # If only prompts file changed, add a generic message part
             unique_details.append("artifacts")
        commit_message = f"Regenerate {', '.join(unique_details)} for {archive_id}"
        logger.info(f"Committing regenerated artifacts to Git: {commit_message}")
        publish_to_git(commit_message)
    else:
        logger.info("No changes detected in Pelican content directory. Skipping Git commit.")

    # --- Cleanup ---
    cleanup_output_directory(temp_regen_dir)
    logger.info(f"--- Regeneration Finished for Archive ID: {archive_id} ---")


# --- New function to publish archived article ---
def publish_archived_article(archive_id: str) -> bool:
    """
    Copies an archived article and its media from the archive directory
    to the Pelican content directory and commits the changes to Git.

    Args:
        archive_id (str): The name of the archive directory (e.g., 'YYYY-MM-DD-slug').

    Returns:
        bool: True if publishing and committing were successful, False otherwise.
    """
    logger.info(f"--- Publishing Archived Article: {archive_id} ---")

    archive_dir = BASE_ARCHIVE_DIR / archive_id
    if not archive_dir.is_dir():
        logger.error(f"Archive directory not found: {archive_dir}")
        return False

    # --- Find and Load Original Data (for metadata and paths) ---
    markdown_files = list(archive_dir.glob("*.md"))
    if not markdown_files:
        logger.error(f"No markdown file found in archive: {archive_dir}")
        return False
    archived_markdown_path = markdown_files[0]
    if len(markdown_files) > 1:
        logger.warning(f"Multiple markdown files found in {archive_dir}. Using: {archived_markdown_path.name}")

    logger.info(f"Loading metadata from archived markdown: {archived_markdown_path.name}")
    metadata = {}
    try:
        with open(archived_markdown_path, 'r', encoding='utf-8') as f:
            markdown_content = f.read()
        metadata, _ = extract_pelican_metadata(markdown_content) # Only need metadata here
    except Exception as e:
        logger.error(f"Failed to read or parse metadata from {archived_markdown_path}: {e}", exc_info=True)
        # Continue if metadata extraction fails, but log the error
        metadata = {}

    # Determine if Denizen based on category metadata
    is_denizen = metadata.get('category', '').lower() == 'dimensional denizen'
    logger.info(f"Article Category: {metadata.get('category', 'Unknown')}, Is Denizen: {is_denizen}")

    # --- Determine Pelican Target Directories ---
    if not PELICAN_CONTENT_DIR_PATH or not Path(PELICAN_CONTENT_DIR_PATH).is_dir():
        logger.error(f"PELICAN_CONTENT_DIR not configured or found: {PELICAN_CONTENT_DIR_PATH}. Cannot publish.")
        return False

    pelican_content_base = Path(PELICAN_CONTENT_DIR_PATH)
    pelican_md_subdir_rel = PELICAN_ARTICLES_SUBDIR if not is_denizen else PELICAN_DENIZENS_SUBDIR
    pelican_md_target_dir = pelican_content_base / pelican_md_subdir_rel
    pelican_img_target_dir = pelican_content_base / PELICAN_IMAGES_SUBDIR
    pelican_audio_target_dir = pelican_content_base / PELICAN_AUDIO_SUBDIR

    # Ensure target directories exist
    try:
        pelican_md_target_dir.mkdir(parents=True, exist_ok=True)
        pelican_img_target_dir.mkdir(parents=True, exist_ok=True)
        pelican_audio_target_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Ensured Pelican target directories exist.")
    except Exception as e:
        logger.error(f"Failed to create Pelican target directories: {e}", exc_info=True)
        return False

    # --- Copy Files to Pelican Content Directory ---
    files_copied = False

    # Copy Markdown
    try:
        target_md_path = pelican_md_target_dir / archived_markdown_path.name
        shutil.copy2(archived_markdown_path, target_md_path)
        logger.info(f"Copied markdown to Pelican: {target_md_path}")
        files_copied = True
    except Exception as e:
        logger.error(f"Failed to copy markdown file {archived_markdown_path.name} to {pelican_md_target_dir}: {e}")
        # Continue attempting to copy other files, but mark failure

    # Copy Images
    archive_images_dir = archive_dir / "images"
    if archive_images_dir.is_dir():
        for img_path in archive_images_dir.glob("*"):
            if img_path.is_file():
                try:
                    target_img_path = pelican_img_target_dir / img_path.name
                    shutil.copy2(img_path, target_img_path)
                    logger.info(f"Copied image to Pelican: {target_img_path}")
                    files_copied = True
                except Exception as e:
                    logger.error(f"Failed to copy image file {img_path.name} to {pelican_img_target_dir}: {e}")
    else:
        logger.warning(f"Archive images directory not found: {archive_images_dir}. No images to copy.")

    # Copy Audio
    archive_audio_dir = archive_dir / "audio"
    if archive_audio_dir.is_dir():
        for audio_path in archive_audio_dir.glob("*"):
             if audio_path.is_file():
                try:
                    target_audio_path = pelican_audio_target_dir / audio_path.name
                    shutil.copy2(audio_path, target_audio_path)
                    logger.info(f"Copied audio to Pelican: {target_audio_path}")
                    files_copied = True
                except Exception as e:
                    logger.error(f"Failed to copy audio file {audio_path.name} to {pelican_audio_target_dir}: {e}")
    else:
        logger.warning(f"Archive audio directory not found: {archive_audio_dir}. No audio to copy.")

    # Note: Video is not copied to Pelican content dir, only the YouTube URL is referenced in markdown.
    # The markdown copy step above handles the markdown content itself, including the URL if present in the archive.

    # --- Commit Changes to Git ---
    if files_copied:
        commit_message = f"Publish archived article: {archive_id}"
        logger.info(f"Committing published article to Git: {commit_message}")
        if publish_to_git(commit_message):
            logger.info("Git commit successful.")
            return True
        else:
            logger.error("Git commit failed.")
            return False
    else:
        logger.warning("No files were successfully copied to Pelican. Skipping Git commit.")
        return False


# --- Function to set article status in Pelican markdown ---
def set_article_status(archive_id: str, status: str) -> bool:
    """
    Finds the Pelican markdown file for an archived article and sets its Status metadata.

    Args:
        archive_id (str): The name of the archive directory (e.g., 'YYYY-MM-DD-slug').
        status (str): The desired status ('draft' or 'published').

    Returns:
        bool: True if the status was updated and committed, False otherwise.
    """
    logger.info(f"--- Setting Status for Article: {archive_id} to '{status}' ---")

    if status.lower() not in ['draft', 'published']:
        logger.error(f"Invalid status '{status}'. Must be 'draft' or 'published'.")
        return False

    if not PELICAN_CONTENT_DIR_PATH or not Path(PELICAN_CONTENT_DIR_PATH).is_dir():
        logger.error(f"PELICAN_CONTENT_DIR not configured or found: {PELICAN_CONTENT_DIR_PATH}. Cannot set status.")
        return False

    # Determine if it's a Denizen to find the correct subdir
    # Read the archived markdown to get the category
    archive_dir = BASE_ARCHIVE_DIR / archive_id
    archived_markdown_path_in_archive = next(archive_dir.glob("*.md"), None)

    is_denizen = False
    if archived_markdown_path_in_archive and archived_markdown_path_in_archive.exists():
        try:
            with open(archived_markdown_path_in_archive, 'r', encoding='utf-8') as f:
                archived_markdown_content = f.read()
            metadata, _ = extract_pelican_metadata(archived_markdown_content) # Only need metadata here
            is_denizen = metadata.get('category', '').lower() == 'dimensional denizen'
            logger.info(f"Determined category from archive: '{metadata.get('category', 'Unknown')}'. Is Denizen: {is_denizen}")
        except Exception as e:
            logger.warning(f"Could not read archived markdown {archived_markdown_path_in_archive} to determine category: {e}. Assuming standard article.")
            is_denizen = False
    else:
        logger.warning(f"Archived markdown file not found for {archive_id}. Assuming standard article for Pelican path.")
        is_denizen = False


    pelican_content_base = Path(PELICAN_CONTENT_DIR_PATH)
    pelican_md_subdir_rel = PELICAN_ARTICLES_SUBDIR if not is_denizen else PELICAN_DENIZENS_SUBDIR
    pelican_md_dir = pelican_content_base / pelican_md_subdir_rel
    pelican_article_path = pelican_md_dir / f"{archive_id}.md"

    if not pelican_article_path.exists():
        logger.error(f"Pelican markdown file not found: {pelican_article_path}")
        # Try the other subdir just in case the category detection was wrong or file was moved
        other_subdir_rel = PELICAN_DENIZENS_SUBDIR if not is_denizen else PELICAN_ARTICLES_SUBDIR
        other_md_dir = pelican_content_base / other_subdir_rel
        other_pelican_article_path = other_md_dir / f"{archive_id}.md"
        if other_pelican_article_path.exists():
             logger.warning(f"File not found in expected subdir, but found in {other_subdir_rel}: {other_pelican_article_path}. Using this path.")
             pelican_article_path = other_pelican_article_path
             is_denizen = not is_denizen # Update denizen status based on found location
        else:
            logger.error(f"Pelican markdown file not found in either standard or denizen subdir for {archive_id}.")
            return False

    try:
        with open(pelican_article_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        new_lines = []
        status_updated = False
        metadata_end_index = -1 # Index where metadata ends (first blank line after metadata)
        status_line_found = False

        # Find existing Status line and the end of metadata
        for i, line in enumerate(lines):
            if line.strip().lower().startswith('status:'):
                new_lines.append(f"Status: {status}\n")
                status_updated = True
                status_line_found = True
                logger.info(f"Found existing Status line, updating to '{status}'.")
            else:
                new_lines.append(line)

            # Check for the end of the metadata block (first blank line)
            if metadata_end_index == -1 and line.strip() == "" and i > 0 and lines[i-1].strip() != "":
                 metadata_end_index = i

        # If the Status line was not found, add it at the end of the metadata block
        if not status_line_found:
            if metadata_end_index != -1:
                # Insert the new line before the content body starts
                new_lines.insert(metadata_end_index, f"Status: {status}\n")
                logger.info(f"Added new Status line: 'Status: {status}'.")
                status_updated = True
            else:
                # Fallback: Add it after the last metadata line if no blank line exists
                last_metadata_line = -1
                for i in range(len(lines)):
                    if re.match(r'^[A-Za-z0-9_-]+:\s*.*$', lines[i].strip()):
                        last_metadata_line = i
                    elif lines[i].strip() == "":
                         # Found a blank line, metadata ends before this
                         break # Exit loop, metadata_end_index should have been set
                    else:
                         # Found content line before a blank line
                         break # Exit loop

                if last_metadata_line != -1:
                    new_lines.insert(last_metadata_line + 1, f"Status: {status}\n")
                    logger.info(f"Added new Status line after last metadata line: 'Status: {status}'.")
                    status_updated = True
                else:
                    logger.warning("Could not find a clear place to insert Status line in Pelican markdown.")


        if status_updated:
            with open(pelican_article_path, 'w', encoding='utf-8') as f:
                f.writelines(new_lines)
            logger.info(f"Successfully updated/added Status to '{status}' in Pelican markdown.")

            # Commit the change
            commit_message = f"Set status of {archive_id} to '{status}'"
            logger.info(f"Committing status change to Git: {commit_message}")
            if publish_to_git(commit_message):
                logger.info("Git commit successful.")
                return True
            else:
                logger.error("Git commit failed.")
                return False
        else:
            logger.warning(f"Status was already '{status}' or could not be updated/added. Skipping Git commit.")
            return False # Status wasn't changed/added

    except Exception as e:
        logger.error(f"Failed to update status in {pelican_article_path}: {e}", exc_info=True)
        return False


# --- Orchestration Function (Implemented) ---
def orchestrate_artifact_generation(
    reporter: Reporter,
    selected_title: str,
    article_content: str, # This is now the clean content body
    summary_text: str,
    topic_text: str, # Not directly used in artifact generation, but kept for context if needed
    filename_base: str,
    current_output_dir: Path,
    publish_date: datetime,
    skip_steps: List[str],
    archiver: Archiver,
    story_data_for_pelican: Dict[str, Any] # Includes location, stardate, featured_characters, etc. (from JSON)
    ) -> Dict[str, Any] | None:
    """
    Orchestrates the generation of all artifacts for a single story.

    Args:
        reporter (Reporter): The reporter object.
        selected_title (str): The chosen title for the article.
        article_content (str): The main body content of the article.
        summary_text (str): The generated summary.
        topic_text (str): The original topic (for context, not direct artifact).
        filename_base (str): Base filename slug (e.g., 'YYYY-MM-DD-slug').
        current_output_dir (Path): Temporary directory for this story's artifacts.
        publish_date (datetime): The target publish date.
        skip_steps (List[str]): List of steps to skip.
        archiver (Archiver): Initialized Archiver instance.
        story_data_for_pelican (Dict[str, Any]): Original story data including location, stardate, etc.

    Returns:
        Dict[str, Any] | None: Dictionary containing paths to generated artifacts and metadata,
                               or None if processing failed critically.
    """
    logger.info(f"--- Orchestrating Artifact Generation for: {selected_title} ---")

    generated_artifacts = {
        "markdown": None,
        "feature_image": None,
        "article_images": [], # List of paths
        "audio": None,
        "video": None,
        "summary_file": None,
        "image_prompts_file": None,
    }
    metadata = {
        "title": selected_title,
        "filename_base": filename_base,
        "archive_path": None,
        "pelican_path": None,
        "youtube_url": None,
        "location": story_data_for_pelican.get('location', 'Unknown Location'),
        "stardate": story_data_for_pelican.get('stardate', 'N/A'),
        "featured_characters": story_data_for_pelican.get('featured_characters', []),
        "category": story_data_for_pelican.get('category', 'Article'), # Default to Article
        "tags": story_data_for_pelican.get('tags', []), # Get tags from input if present
    }

    # Ensure tags include reporter tags and 'The Ephergent'
    reporter_tags = reporter.tags if reporter and reporter.tags else []
    base_tags = ["The Ephergent"]
    # Combine and deduplicate tags, ensuring they are strings
    all_tags = list(set([str(tag) for tag in (metadata['tags'] + reporter_tags + base_tags) if tag]))
    metadata['tags'] = all_tags
    logger.info(f"Using tags: {metadata['tags']}")


    # Prepare story_data dict for image generation (needs title, content, location, stardate, featured_characters)
    story_data_for_images = {
        'title': selected_title,
        'content': article_content,
        'location': metadata['location'],
        'stardate': metadata['stardate'],
        'featured_characters': metadata['featured_characters'],
    }
    image_prompt_details = [] # List to store prompt details for JSON file

    # 1. Generate Images
    if "image" not in skip_steps:
        logger.info("Generating Images...")
        temp_img_dir = current_output_dir / "images"
        temp_img_dir.mkdir(exist_ok=True)

        # Generate Feature Image
        feature_image_filename = f"{filename_base}_feature.png"
        feature_image_path = generate_ephergent_image(
            story_data=story_data_for_images, reporter=reporter, image_type="featured",
            output_dir=temp_img_dir, filename=feature_image_filename,
            prompt_details_list=image_prompt_details # Pass list to store prompt
        )
        if feature_image_path:
            generated_artifacts["feature_image"] = feature_image_path
            logger.info(f"Generated feature image: {feature_image_path.name}")
        else:
            logger.warning("Failed to generate feature image.")

        # Generate Article Essence Image (single image)
        article_essence_prompt = generate_article_essence_image_prompt(
            story_data=story_data_for_images,
            reporter=reporter
        )
        if article_essence_prompt:
            article_image_filename = f"{filename_base}_article_essence.png"
            article_image_path = generate_ephergent_image(
                story_data=story_data_for_images, reporter=reporter, image_type="article",
                output_dir=temp_img_dir, filename=article_image_filename, image_index=0, # Index 0 for the single image
                prompt_override=article_essence_prompt, # Use the pre-generated prompt
                prompt_details_list=image_prompt_details # Pass list to store prompt
            )
            if article_image_path:
                generated_artifacts["article_images"] = [article_image_path] # Store as a list
                logger.info(f"Generated article essence image: {article_image_path.name}")
            else:
                logger.warning("Failed to generate article essence image.")
        else:
            logger.warning("Failed to generate article essence image prompt.")


        # Save image prompts JSON
        if image_prompt_details:
            prompts_json_path = current_output_dir / f"{filename_base}_image_prompts.json"
            try:
                with open(prompts_json_path, 'w', encoding='utf-8') as f:
                    json.dump(image_prompt_details, f, indent=2)
                generated_artifacts["image_prompts_file"] = prompts_json_path
                logger.info(f"Saved image prompts details to: {prompts_json_path.name}")
            except Exception as e:
                logger.error(f"Failed to save image prompts JSON: {e}")

    else:
        logger.info("Skipping Image Generation.")


    # 2. Generate Audio
    audio_path = None
    if "audio" not in skip_steps:
        logger.info("Generating Audio...")
        # For standard articles, use the main content body
        audio_path = generate_article_audio(
            reporter=reporter, article_content=article_content, title=selected_title,
            output_dir=current_output_dir, filename_base=filename_base, speed=1.1 # Use base output dir for audio
        )
        if audio_path:
            generated_artifacts["audio"] = audio_path
            logger.info(f"Generated audio: {audio_path.name}")
        else:
            logger.warning("Failed to generate audio.")
    else:
        logger.info("Skipping Audio Generation.")

    # 3. Generate Video
    video_path = None
    # Video requires audio and at least a feature image
    if "video" not in skip_steps and generated_artifacts.get("audio") and generated_artifacts.get("feature_image"):
        logger.info("Generating Video...")
        temp_video_dir = current_output_dir / "video"
        temp_video_dir.mkdir(exist_ok=True)
        video_path = generate_youtube_video(
            reporter=reporter, title=selected_title, audio_path=generated_artifacts["audio"],
            featured_image_path=generated_artifacts["feature_image"],
            article_image_paths=generated_artifacts["article_images"], # Pass the list (will be 0 or 1 item)
            output_dir=temp_video_dir, filename_base=filename_base
        )
        if video_path:
            generated_artifacts["video"] = video_path
            logger.info(f"Generated video: {video_path.name}")
        else:
            logger.warning("Failed to generate video.")
    else:
        if "video" not in skip_steps: # Only log warning if video was requested but inputs missing
             logger.warning("Skipping Video Generation (requires audio and feature image).")
        else:
             logger.info("Skipping Video Generation.")


    # 4. Format Pelican Markdown
    logger.info("Formatting Pelican Markdown...")
    # Determine category and tags for Pelican metadata
    category = metadata.get('category', 'Article') # Use category from input data, default to Article
    tags = metadata.get('tags', []) # Use tags from input data, combined with reporter/base tags earlier

    # Prepare the story_data dictionary for format_story_markdown
    # This should contain the final, processed values
    story_data_for_markdown = {
        'title': selected_title,
        'content': article_content, # Use the clean content body
        'summary': summary_text,
        'location': metadata['location'], # Use the determined location
        'stardate': metadata['stardate'], # Use the generated stardate
        'featured_characters': metadata['featured_characters'], # Use the determined featured characters
        # Add other relevant fields if format_story_markdown uses them from story_data
        # e.g., 'Month', 'Week', 'Filed by' might be useful for the template
        'Month': story_data_for_pelican.get('Month'), # Keep original Month/Week if available
        'Week': story_data_for_pelican.get('Week'),
        'Filed by': reporter.name, # Use the reporter name
    }

    # Correctly call format_story_markdown with story_data dict and other explicit args
    pelican_markdown_content = format_story_markdown(
        story_data=story_data_for_markdown, # Pass the constructed dict as the first argument
        reporter_name=reporter.name,        # Pass reporter name explicitly
        category=category,                  # Pass category explicitly
        tags=tags,                          # Pass tags explicitly
        feature_image_filename=generated_artifacts["feature_image"].name if generated_artifacts.get("feature_image") else None,
        article_image_filenames=[img.name for img in generated_artifacts["article_images"]],
        audio_filename=generated_artifacts["audio"].name if generated_artifacts.get("audio") else None,
        youtube_video_url=None, # Still None at this stage
        publish_date=publish_date
    )
    markdown_temp_path = current_output_dir / f"{filename_base}.md"
    with open(markdown_temp_path, "w", encoding="utf-8") as f:
        f.write(pelican_markdown_content)
    generated_artifacts["markdown"] = markdown_temp_path
    logger.info(f"Formatted Pelican markdown: {markdown_temp_path.name}")


    # 5. Export to Pelican Content Directory
    pelican_path = None
    if "export" not in skip_steps:
        logger.info("Exporting to Pelican...")
        media_for_export = {
            "feature_image": generated_artifacts.get("feature_image"),
            "article_images": generated_artifacts["article_images"], # Pass the list
            "audio": generated_artifacts.get("audio")
        }
        # Determine if it's a denizen for export path
        is_denizen = metadata.get('category', '').lower() == 'dimensional denizen'

        pelican_path = export_to_pelican(
            markdown_content=pelican_markdown_content,
            markdown_filename_base=filename_base,
            media_paths=media_for_export,
            is_denizen=is_denizen # Pass the denizen status
        )
        if pelican_path:
            metadata['pelican_path'] = pelican_path
            logger.info(f"Exported to Pelican: {pelican_path}")
        else:
            logger.error("Failed to export to Pelican.")
    else:
        logger.info("Skipping Pelican Export.")


    # 6. Archive Artifacts
    archive_dir = None
    if "archive" not in skip_steps and archiver and archiver.initialized:
        logger.info("Archiving Artifacts...")
        # Create a summary file to archive
        summary_path = current_output_dir / f"{filename_base}_summary.txt"
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(summary_text)
        generated_artifacts["summary_file"] = summary_path

        # Pass all generated artifact paths to the archiver
        archive_dir = archiver.archive_artifacts(filename_base, generated_artifacts)
        if archive_dir:
            metadata['archive_path'] = archive_dir
            logger.info(f"Archived artifacts to: {archive_dir}")
        else:
            logger.warning("Archiving failed.")
    else:
        logger.info("Skipping Archiving.")


    # 7. Upload to YouTube
    youtube_url = None
    # YouTube upload requires the generated video file
    if "youtube" not in skip_steps and generated_artifacts.get("video"):
        logger.info("Uploading to YouTube...")
        # Use summary as video description initially, add link later if Pelican export was successful
        video_description = summary_text
        if metadata.get('pelican_path'):
             # Construct the final URL based on the Pelican path and BLOG_URL
             # Assuming pelican_path is relative to PELICAN_CONTENT_DIR_PATH
             relative_pelican_path = Path(metadata['pelican_path']).relative_to(PELICAN_CONTENT_DIR_PATH)
             article_url = f"{BLOG_URL}/{relative_pelican_path.parent.name}/{relative_pelican_path.name.replace('.md', '.html')}"
             video_description += f"\n\nRead the full report: {article_url}"
        else:
             video_description += "\n\nRead the full report: [Link will be available on blog soon]"

        # Use tags from metadata for YouTube tags
        video_tags = metadata.get('tags', [])

        youtube_video_id = upload_to_youtube(
            video_file_path=generated_artifacts["video"],
            title=selected_title,
            description=video_description,
            tags=video_tags,
            thumbnail_path=generated_artifacts.get("feature_image") # Use feature image as thumbnail
        )
        if youtube_video_id:
            youtube_url = f"https://www.youtube.com/watch?v={youtube_video_id}"
            metadata['youtube_url'] = youtube_url
            logger.info(f"YouTube upload successful: {youtube_url}")

            # Update Pelican markdown with YouTube URL if export was successful
            if metadata.get('pelican_path'):
                 pelican_article_path = Path(metadata['pelican_path'])
                 if pelican_article_path.exists():
                     try:
                         with open(pelican_article_path, 'r', encoding='utf-8') as f:
                             lines = f.readlines()
                         new_lines = []
                         youtube_metadata_added = False
                         youtube_embed_added = False
                         metadata_end_index = -1
                         content_start_index = -1

                         # Find metadata end and content start
                         for i, line in enumerate(lines):
                             if metadata_end_index == -1 and line.strip() == "" and i > 0 and lines[i-1].strip() != "":
                                 metadata_end_index = i
                                 content_start_index = i + 1
                             new_lines.append(line) # Copy all lines initially

                         # Add YouTube URL metadata if not present
                         if not any(line.strip().lower().startswith('youtube_url:') or line.strip().lower().startswith('youtubeurl:') for line in new_lines[:metadata_end_index if metadata_end_index != -1 else len(new_lines)]):
                             insert_index = metadata_end_index if metadata_end_index != -1 else len(new_lines)
                             new_lines.insert(insert_index, f"YouTubeURL: {youtube_url}\n")
                             logger.info("Added YouTubeURL metadata to Pelican markdown.")
                             youtube_metadata_added = True
                             if content_start_index != -1: content_start_index += 1 # Adjust content start index

                         # Add YouTube embed HTML to content body if not present
                         if not any('<iframe' in line and 'youtube.com/embed' in line for line in new_lines[content_start_index:] if content_start_index != -1):
                             video_id_match = re.search(r"v=([^&]+)", youtube_url)
                             video_id = video_id_match.group(1) if video_id_match else None
                             if video_id:
                                 new_embed_html_lines = [
                                     '<figure>\n',
                                     '  <div style="position: relative; padding-bottom: 56.25%; height: 0; overflow: hidden; max-width: 100%; height: auto;">\n',
                                     f'    <iframe src="https://www.youtube.com/embed/{video_id}" style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>\n',
                                     '  </div>\n',
                                     '  <figcaption style="font-size: 0.8em; color: grey;">⁂ Video created by The Ephergent\'s dimensionally-aware AI ⁂</figcaption>\n',
                                     '</figure>\n'
                                 ]
                                 # Find insertion point: before '---' or at the end of content body
                                 insertion_index = len(new_lines) # Default to end
                                 if content_start_index != -1:
                                     for i in range(content_start_index, len(new_lines)):
                                         if new_lines[i].strip() == '---':
                                             insertion_index = i
                                             break # Insert before the horizontal rule
                                 else:
                                     logger.warning("Could not determine content start index to insert YouTube embed HTML.")

                                 if insertion_index != len(new_lines) or content_start_index != -1: # Only insert if we found a place or content started
                                     new_lines[insertion_index:insertion_index] = new_embed_html_lines
                                     logger.info("Added YouTube embed HTML to Pelican markdown.")
                                     youtube_embed_added = True
                                 else:
                                     logger.warning("Could not find a suitable insertion point for YouTube embed HTML.")


                         if youtube_metadata_added or youtube_embed_added:
                             with open(pelican_article_path, 'w', encoding='utf-8') as f:
                                 f.writelines(new_lines)
                             logger.info("Successfully updated Pelican markdown with YouTube info.")
                         else:
                             logger.info("Pelican markdown already contained YouTube info or no info to add.")

                     except Exception as e:
                         logger.error(f"Failed to update Pelican markdown with YouTube URL/embed: {e}", exc_info=True)
                 else:
                     logger.warning(f"Pelican markdown file not found at {pelican_article_path}, cannot update with YouTube URL.")
            else:
                 logger.warning("Pelican export skipped or failed, cannot update markdown with YouTube URL.")

        else:
            logger.warning("YouTube upload failed.")
    else:
        if "youtube" not in skip_steps: # Only log warning if YouTube was requested but inputs missing
             logger.warning("Skipping YouTube Upload (requires video).")
        else:
             logger.info("Skipping YouTube Upload.")


    # 8. Publish to Git (if Pelican export was successful)
    if "git" not in skip_steps and metadata.get('pelican_path'):
        logger.info("Publishing draft to Git...")
        # The export_to_pelican function already copies files to the Pelican content dir.
        # We just need to commit those changes.
        commit_message = f"Add draft article: {selected_title}"
        publish_to_git(commit_message)
    else:
        if "git" not in skip_steps: # Only log warning if Git was requested but Pelican export missing
             logger.warning("Skipping Git Publish (requires successful Pelican export).")
        else:
             logger.info("Skipping Git Publish.")


    # 9. Post to Social Media (if Pelican export was successful)
    if "social" not in skip_steps and metadata.get('pelican_path'):
        logger.info("Posting to Social Media...")
        try:
            # Construct the final URL based on the Pelican path and BLOG_URL
            relative_pelican_path = Path(metadata['pelican_path']).relative_to(PELICAN_CONTENT_DIR_PATH)
            article_url = f"{BLOG_URL}/{relative_pelican_path.parent.name}/{relative_pelican_path.name.replace('.md', '.html')}"

            post_article_to_social_media(
                title=selected_title,
                url=article_url,
                summary=summary_text,
                image_path=generated_artifacts.get("feature_image"), # Use feature image
                tags=metadata.get('tags', []) # Use tags from metadata
            )
        except Exception as e:
            logger.error(f"Failed to post to social media: {e}", exc_info=True)
    else:
        if "social" not in skip_steps: # Only log warning if Social was requested but Pelican export missing
             logger.warning("Skipping Social Media Post (requires successful Pelican export).")
        else:
             logger.info("Skipping Social Media Post.")


    # 10. Send Email Notification (if Pelican export was successful)
    if "mail" not in skip_steps and metadata.get('pelican_path'):
        logger.info("Sending Email Notification...")
        try:
            # Construct the final URL based on the Pelican path and BLOG_URL
            relative_pelican_path = Path(metadata['pelican_path']).relative_to(PELICAN_CONTENT_DIR_PATH)
            article_url = f"{BLOG_URL}/{relative_pelican_path.parent.name}/{relative_pelican_path.name.replace('.md', '.html')}"

            # Use feature image URL for email template
            feature_image_url = f"{BLOG_URL}/{PELICAN_IMAGES_SUBDIR}/{generated_artifacts['feature_image'].name}" if generated_artifacts.get("feature_image") else f"{BLOG_URL}/theme/images/ephergent_logo.png"

            email_vars = {
                "newsletter_date": publish_date.strftime("%B %d, %Y"),
                "article_title": selected_title,
                "article_summary": summary_text,
                "article_url": article_url,
                "article_feature_image_url": feature_image_url,
            }
            send_email(
                subject=f"The Ephergent: {selected_title}",
                template=os.getenv('MAILGUN_TEMPLATE', "daily.dimensional.dispatch"), # Use configurable template
                variables=email_vars
            )
        except Exception as e:
            logger.error(f"Failed to send email: {e}", exc_info=True)
    else:
        if "mail" not in skip_steps: # Only log warning if Mail was requested but Pelican export missing
             logger.warning("Skipping Email Notification (requires successful Pelican export).")
        else:
             logger.info("Skipping Email Notification.")


    logger.info(f"--- Artifact Generation Orchestration Complete for: {selected_title} ---")

    # Return metadata and artifact paths
    return {
        "metadata": metadata,
        "artifacts": generated_artifacts
    }


# --- Main Workflow for Processing a Single Story (from input file) ---
def process_story_from_file(
    story_data: Dict[str, Any],
    story_file_path: Path, # Add path to the original input file
    archiver: Archiver, # Pass archiver instance
    skip_steps: Optional[List[str]] = None
    ) -> Dict[str, Any] | None:
    """
    Processes a single story from a JSON input file, generating all artifacts.

    Args:
        story_data (Dict[str, Any]): The dictionary representing one story.
        story_file_path (Path): The path to the original input JSON file.
        archiver (Archiver): Initialized Archiver instance.
        skip_steps (Optional[List[str]]): List of steps to skip for this run.

    Returns:
        Dict[str, Any] | None: Dictionary containing paths to generated artifacts and metadata,
                               or None if processing failed critically.
    """
    process_start_time = datetime.now()
    if skip_steps is None:
        skip_steps = SKIP_STEPS # Use global default if not provided

    # Extract data from the story_data dictionary loaded from the JSON file
    # Assume the JSON file structure is correct (metadata keys separate from content body)
    title = story_data.get('title', f"Untitled_{process_start_time.strftime('%Y%m%d%H%M%S')}")
    filed_by = story_data.get("Filed by", "Pixel Paradox")
    article_content = story_data.get('content', '') # This should be the clean body from JSON
    summary_text = story_data.get('summary', '')
    topic_text = story_data.get('topic', title) # Use topic from file or title
    month = story_data.get("Month")
    week = story_data.get("Week")
    # location and stardate are also expected to be in story_data

    logger.info(f"--- Processing Story from File: {title} from {story_file_path.name} ---")

    # --- Setup for this story ---
    run_timestamp = process_start_time.strftime('%Y%m%d_%H%M%S')
    safe_title_slug = sanitize_filename(title)
    # Use the date prefix from the calculated publish date for the filename_base
    # This ensures consistency between the input filename (MMM_WW) and the output slug (YYYY-MM-DD-slug)
    # Calculate publish date first to get the date prefix
    publish_date = None
    if isinstance(month, int) and isinstance(week, int):
        publish_date = calculate_publish_date(month, week, SEASON_START_DATE)
    else:
        logger.warning(f"Missing or invalid Month/Week for story '{title}'. Using current date for export.")
        publish_date = process_start_time # Fallback to now

    date_prefix = publish_date.strftime('%Y-%m-%d')
    filename_base = f"{date_prefix}-{safe_title_slug}"

    current_output_dir = OUTPUT_DIR / f"story_run_{filename_base}" # Use filename_base in temp dir name
    current_output_dir.mkdir(exist_ok=True)
    logger.info(f"Using temporary output directory for this story: {current_output_dir}")

    try:
        # --- Get Reporter ---
        reporter = get_reporter_from_name(filed_by)
        if not reporter:
            logger.error(f"Could not get reporter for '{filed_by}'. Skipping story.")
            return None

        # --- Orchestrate Artifact Generation ---
        # Pass the extracted content body and the full story_data dict
        processed_data = orchestrate_artifact_generation(
            reporter=reporter,
            selected_title=title,
            article_content=article_content, # Pass the clean content body
            summary_text=summary_text,
            topic_text=topic_text,
            filename_base=filename_base,
            current_output_dir=current_output_dir,
            publish_date=publish_date,
            skip_steps=skip_steps,
            archiver=archiver,
            story_data_for_pelican=story_data # Pass the full story_data dict (includes location, stardate, etc.)
        )

        if processed_data:
            # --- Move Input File to Archived State (only if from file and successful) ---
            # Note: This move only happens if the 'archive' step was NOT skipped.
            # If 'archive' is skipped, the file remains in 'ready'.
            # Check if the archive_path was successfully set in the returned metadata
            archive_success = processed_data.get("metadata", {}).get("archive_path") is not None
            if story_file_path and archive_success:
                try:
                    # Ensure the archive directory for the *input file* exists
                    ARCHIVED_STORIES_DIR.mkdir(parents=True, exist_ok=True)
                    new_path = ARCHIVED_STORIES_DIR / story_file_path.name
                    shutil.move(story_file_path, new_path)
                    logger.info(f"Moved input story file to archived: {new_path}")
                except Exception as e:
                    logger.error(f"Failed to move input story file {story_file_path} to archived: {e}", exc_info=True)
            elif story_file_path:
                # If archive was skipped, the file stays in 'ready'.
                # We don't need to do anything here, just log that it wasn't moved.
                logger.info(f"Input file {story_file_path.name} remains in 'ready' directory (archive step skipped or failed).")

            logger.info(f"--- Finished Processing Story from File: {title} ---")
            return processed_data
        else:
            logger.error(f"--- Failed Processing Story from File: {title} ---")
            return None # Indicate failure
    except Exception as e:
        logger.error(f"Unexpected error during processing story from file '{title}': {e}", exc_info=True)
        return None
    finally:
        # Cleanup of current_output_dir is handled here
        cleanup_output_directory(current_output_dir)


# --- New Generative Workflow ---
def run_generative_workflow(
    reporter_id_arg: Optional[str],
    topic_arg: Optional[str],
    skip_steps: List[str],
    archiver: Archiver
    ) -> None:
    """
    Runs the fully automatic content generation workflow.
    Generates content, saves it to a JSON file in 'ready', and then processes the file.
    """
    logger.info("--- Starting New Generative Workflow ---")
    process_start_time = datetime.now()

    # --- Setup for this run ---
    # Temporary output directory creation/cleanup is now handled by process_story_from_file
    # current_output_dir = OUTPUT_DIR / f"generated_run_{run_timestamp}"
    # current_output_dir.mkdir(exist_ok=True)
    # logger.info(f"Using temporary output directory for this run: {current_output_dir}")

    try:
        # 1. Select Reporter
        if reporter_id_arg:
            reporter = Reporter(identifier=reporter_id_arg)
            if not reporter.reporter_data:
                logger.error(f"Reporter ID '{reporter_id_arg}' not found. Exiting generative workflow.")
                return
        else:
            reporter = Reporter.get_random_reporter()
            if not reporter.reporter_data:
                logger.error("Failed to select a random reporter. Exiting generative workflow.")
                return
        logger.info(f"Selected Reporter: {reporter.name} (ID: {reporter.id})")

        # 2. Generate Topic
        if topic_arg:
            topic = topic_arg
            logger.info(f"Using provided topic: {topic}")
        else:
            topic = generate_topic(reporter)
            if not topic:
                logger.error("Failed to generate topic. Exiting generative workflow.")
                return
            logger.info(f"Generated topic: {topic}")

        # 3. Vector DB Search for similar articles (Contextual Awareness)
        similar_articles_context = []
        if archiver and archiver.initialized and "archive" not in skip_steps : # only search if archiver is good and not skipping archive
            logger.info(f"Searching for articles similar to topic: '{topic}'")
            try:
                similar_articles_context = archiver.search_similar_articles(query_text=topic, top_k=3)
                if similar_articles_context:
                    logger.info(f"Found {len(similar_articles_context)} similar articles for context.")
                else:
                    logger.info("No similar articles found for context.")
            except Exception as e:
                logger.warning(f"Error searching for similar articles: {e}")
        else:
            logger.info("Skipping similar article search (Archiver not available or archive step skipped).")


        # 4. Generate Article Content (Raw LLM Output)
        logger.info("Generating article content...")
        # This returns a dictionary with 'content' (raw text), 'featured_characters' (LLM's list), 'stardate' (LLM's stardate)
        article_data_raw = generate_article(reporter, topic, similar_articles_context=similar_articles_context)
        if not article_data_raw or not article_data_raw.get('content'):
            logger.error("Failed to generate article content. Exiting generative workflow.")
            return

        raw_article_text = article_data_raw['content']
        llm_featured_characters = article_data_raw.get('featured_characters', [])
        llm_stardate = article_data_raw.get('stardate', 'N/A') # Keep LLM's stardate just in case, though we generate our own

        # 5. Parse Raw Article Content
        logger.info("Parsing raw article content to extract metadata and body...")
        try:
            parsed_metadata, content_body = extract_pelican_metadata(raw_article_text)
            logger.info("Successfully parsed article content.")
            # Use parsed metadata, but prefer programmatically generated values for location/stardate
            parsed_title = parsed_metadata.get('title', 'Untitled Report')
            parsed_filed_by = parsed_metadata.get('author', reporter.name) # LLM might put author in 'Filed by' or 'Author'
            parsed_featured_characters = parsed_metadata.get('featured_characters', llm_featured_characters) # Prefer parsed list if available
            # Ignore parsed_metadata.get('location') and parsed_metadata.get('stardate') here
        except Exception as e:
            logger.error(f"Failed to parse article content metadata: {e}. Using raw content and fallback metadata.", exc_info=True)
            # Fallback: Use raw text as content body and basic metadata
            content_body = raw_article_text
            parsed_title = raw_article_text.split('\n', 1)[0].replace('Title:', '').strip() if raw_article_text else 'Untitled Report'
            parsed_filed_by = reporter.name
            parsed_featured_characters = llm_featured_characters # Use LLM's list if parsing failed


        # 6. Generate Summary (from the clean content body)
        logger.info("Generating summary...")
        summary_text = generate_summary(reporter, content_body)
        if not summary_text:
            logger.warning("Failed to generate summary. Using first part of content as fallback.")
            summary_text = content_body[:150] + "..." if content_body else "No summary available."

        # 7. Generate Title (from the clean content body and topic)
        logger.info("Generating titles...")
        titles_list = generate_titles(reporter, topic, content_body)
        if not titles_list:
            logger.warning("Failed to generate titles. Using parsed title or topic as fallback.")
            selected_title = parsed_title if parsed_title != 'Untitled Report' else topic.title()
        else:
            selected_title = random.choice(titles_list) # Pick a random generated title
        logger.info(f"Selected Title: {selected_title}")

        # --- Determine Next Story ID and Calculate Publish Date ---
        next_month, next_week = get_next_story_id()
        # Calculate publish date based on the determined ID
        publish_date = calculate_publish_date(next_month, next_week, SEASON_START_DATE)
        logger.info(f"Calculated publish date based on ID {next_month:03d}_{next_week:02d}: {publish_date.strftime('%Y-%m-%d')}")

        # --- Generate Ephergent Stardate for Metadata ---
        # Format: Cycle MMM.WWW.DDD (Month.Week.Day)
        day_of_week = random.randint(1, 7) # Pick a random day within the week
        ephergent_stardate = f"Cycle {next_month:03d}.{next_week:03d}.{day_of_week:03d}"
        logger.info(f"Generated Ephergent Stardate for metadata: {ephergent_stardate}")


        # 8. Determine Primary Dimension/Location for Metadata
        # Filter reporter tags to find those that are also core dimensions
        reporter_dimension_tags = [tag for tag in reporter.tags if tag in CORE_DIMENSIONS]

        if reporter_dimension_tags:
            # If reporter has specific dimension tags, pick one randomly
            primary_dimension = random.choice(reporter_dimension_tags)
            logger.info(f"Selected primary dimension from reporter tags: {primary_dimension}")
        else:
            # If reporter has no specific dimension tags, pick one from the core dimensions list
            primary_dimension = random.choice(CORE_DIMENSIONS)
            logger.info(f"Reporter has no specific dimension tags. Selected primary dimension from core dimensions: {primary_dimension}")


        # --- Determine Target Filename ---
        target_filename = f"{next_month:03d}_{next_week:02d}.json"
        target_file_path = INPUT_STORIES_DIR_READY / target_filename

        # Construct the story_data_dict with the correct, flat structure
        story_data_dict = {
            "Month": next_month,
            "Week": next_week,
            "title": selected_title, # Use the selected title
            "Filed by": parsed_filed_by, # Use the parsed author name
            "content": content_body, # Use the extracted content body
            "summary": summary_text,
            "location": primary_dimension, # Use the programmatically determined dimension
            "stardate": ephergent_stardate, # Use the programmatically generated Ephergent stardate
            "featured_characters": parsed_featured_characters # Use the parsed featured characters
        }

        logger.info(f"Saving generated story data to input file: {target_file_path}")
        try:
            INPUT_STORIES_DIR_READY.mkdir(parents=True, exist_ok=True) # Ensure dir exists
            with open(target_file_path, 'w', encoding='utf-8') as f:
                json.dump(story_data_dict, f, indent=2)
            logger.info(f"Successfully saved generated story to {target_file_path.name}")
        except IOError as e:
            logger.error(f"Fatal: Could not save generated story to {target_file_path}: {e}", exc_info=True)
            return # Exit if we can't save the input file

        # --- Process the Newly Created File ---
        # The process_story_from_file function will handle the rest of the steps
        # and will NOT move the file from 'ready' if 'archive' is in skip_steps.
        logger.info(f"Processing the newly created input file: {target_file_path.name}")
        # Pass the constructed story_data_dict directly, as it's already in the desired format
        # Note: process_story_from_file expects the story_data dict AND the file path.
        # It will reload the data from the file path internally. This is slightly redundant
        # but maintains the existing function signature and flow.
        # A cleaner approach might be to pass the dict and *optionally* the file path for archiving.
        # For now, stick to the existing signature.
        # Reloading the data from the file ensures we are processing exactly what was saved.
        try:
            with open(target_file_path, 'r', encoding='utf-8') as f:
                 reloaded_story_data = json.load(f)
            # Add Month/Week back as they are derived from filename, not always in JSON
            match = re.match(r"(\d{3})_(\d{0,2})\.json", target_file_path.name)
            if match:
                 reloaded_story_data['Month'] = int(match.group(1))
                 try:
                     reloaded_story_data['Week'] = int(match.group(2)) if match.group(2) else 0
                 except ValueError:
                     reloaded_story_data['Week'] = 0
            else:
                 reloaded_story_data['Month'] = 0
                 reloaded_story_data['Week'] = 0
                 logger.warning(f"Could not parse Month/Week from generated filename {target_file_path.name}.")


            process_story_from_file(reloaded_story_data, target_file_path, main_archiver, steps_to_skip_list)

        except Exception as e:
             logger.error(f"Error reloading or processing newly created file {target_file_path.name}: {e}", exc_info=True)


        logger.info(f"--- Generative Workflow Completed for: {selected_title} (Processed from file) ---")

    except Exception as e:
        logger.error(f"An error occurred during the generative workflow: {e}", exc_info=True)
    finally:
        # Cleanup is now handled by process_story_from_file
        pass


# --- Main Execution Logic ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ephergent Content Creation System")
    parser.add_argument(
        "--auto-generate",
        action="store_true",
        help="Run the new fully automatic content generation workflow."
    )
    parser.add_argument(
        "--reporter",
        type=str,
        default=None,
        help="Specify reporter ID for the --auto-generate workflow. Random if not set."
    )
    parser.add_argument(
        "--topic",
        type=str,
        default=None,
        help="Specify topic for the --auto-generate workflow. Generated if not set."
    )
    parser.add_argument(
        "--denizen",
        action="store_true",
        help="Run the 'Today's Dimensional Denizen' generation workflow."
    )
    parser.add_argument(
        "--skip",
        type=str,
        default=os.getenv('SKIP_STEPS', ''),
        help="Comma-separated list of steps to skip (e.g., audio,git,youtube,archive,export,image,video,social)."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit processing from 'ready' directory to the first N stories (0 for all)."
    )
    parser.add_argument(
        "--build-memory",
        action="store_true",
        help="Scan the archive directory and build/update the vector database memory."
    )
    parser.add_argument(
        "--regenerate",
        type=str,
        metavar='ARCHIVE_ID',
        help="Regenerate artifacts for a specific archive ID. Requires --steps."
    )
    parser.add_argument(
        "--steps",
        type=str,
        metavar='STEPS',
        help="Comma-separated list of steps to regenerate. Used only with --regenerate."
    )
    parser.add_argument(
        "--publish-archive",
        type=str,
        metavar='ARCHIVE_ID',
        help="Publish a previously archived article to the Pelican content directory and commit to Git."
    )
    parser.add_argument(
        "--set-status",
        type=str,
        metavar='ARCHIVE_ID',
        help="Set the status ('draft' or 'published') for a specific archived article in the Pelican content directory."
    )
    parser.add_argument(
        "--status",
        type=str,
        choices=['draft', 'published'],
        help="The status to set ('draft' or 'published'). Required with --set-status."
    )


    args = parser.parse_args()

    # --- Argument Validation ---
    steps_to_run_list = [] # Initialize to ensure it's defined
    if args.regenerate:
        if not args.steps:
            parser.error("--regenerate requires --steps to specify which steps to run.")
        if args.skip or args.limit or args.denizen or args.build_memory or args.auto_generate or args.publish_archive or args.set_status or args.status:
            parser.error("--regenerate cannot be used with other major modes or --set-status/--status.")
        steps_to_run_list = [step.strip().lower() for step in args.steps.split(',') if step.strip()]
        invalid_steps = [s for s in steps_to_run_list if s not in VALID_REGEN_STEPS]
        if invalid_steps:
            parser.error(f"Invalid steps for regeneration: {', '.join(invalid_steps)}. Valid steps are: {', '.join(VALID_REGEN_STEPS)}")
        # Ensure 'image' implies both feature and article images for regeneration logic
        if 'image' in steps_to_run_list:
             if 'feature_image' not in steps_to_run_list: steps_to_run_list.append('feature_image')
             if 'article_image' not in steps_to_run_list: steps_to_run_list.append('article_image')

    elif args.steps:
        parser.error("--steps can only be used with --regenerate.")
    elif args.publish_archive:
         if args.skip or args.limit or args.denizen or args.build_memory or args.auto_generate or args.regenerate or args.steps or args.set_status or args.status:
             parser.error("--publish-archive cannot be used with other major modes or --set-status/--status.")
    elif args.set_status:
         if not args.status:
             parser.error("--set-status requires --status to specify the desired status ('draft' or 'published').")
         if args.skip or args.limit or args.denizen or args.build_memory or args.auto_generate or args.regenerate or args.steps or args.publish_archive:
             parser.error("--set-status cannot be used with other major modes.")
    elif args.status:
         parser.error("--status can only be used with --set-status.")


    # --- Initialize Archiver (once for the whole run) ---
    # Archiver is needed for build-memory, auto-generate (search), and potentially archive steps
    # It's NOT strictly needed for regenerate, publish-archive, or set-status
    # Initialize it only if a mode that uses it is selected, or if no mode is selected (default processing)
    main_archiver = None
    # Initialize if auto-generate, build-memory, or default processing (no specific mode)
    if args.build_memory or args.auto_generate or (not args.regenerate and not args.denizen and not args.publish_archive and not args.set_status):
        try:
            main_archiver = Archiver()
            logger.info("Main Archiver initialized successfully.")
        except Exception as e:
            logger.error(f"Fatal Error: Failed to initialize Archiver: {e}. Vector DB features disabled.", exc_info=True)
            main_archiver = None
    else:
         logger.info("Skipping Archiver initialization (not required for selected mode).")


    # --- Handle --build-memory flag ---
    if args.build_memory:
        if not main_archiver or not main_archiver.initialized:
             logger.error("Cannot build memory: Archiver failed to initialize.")
        else:
             try:
                 logger.warning("Vector memory building (full rescan) not yet implemented in Archiver. Skipping.")
                 # Placeholder for future implementation:
                 # main_archiver.build_full_vector_memory(BASE_ARCHIVE_DIR)
                 pass
             except Exception as build_e:
                 logger.error(f"Error during vector memory build: {build_e}", exc_info=True)
        logger.info("Exiting after vector memory build attempt.")
        sys.exit(0)

    # --- Handle --regenerate flag ---
    if args.regenerate:
        regenerate_artifacts(args.regenerate, steps_to_run_list)
        logger.info("App execution finished (Regeneration Mode).")
        sys.exit(0)

    # --- Handle --publish-archive flag ---
    if args.publish_archive:
        publish_archived_article(args.publish_archive)
        logger.info("App execution finished (Publish Archive Mode).")
        sys.exit(0)

    # --- Handle --set-status flag ---
    if args.set_status:
        set_article_status(args.set_status, args.status)
        logger.info("App execution finished (Set Status Mode).")
        sys.exit(0)


    # --- Standard Workflow (Initial Processing or Denizen) ---
    steps_to_skip_list = []
    if args.skip:
        steps_to_skip_list = [step.strip().lower() for step in args.skip.split(',') if step.strip()]
        logger.info(f"Requesting to skip steps: {steps_to_skip_list}")

    if args.denizen:
        # --- Run Denizen Profile Generation Workflow ---
        logger.info("Starting Denizen Profile Generation Workflow...")
        if not all([generate_denizen_profile, prepare_denizen_text_for_tts]): # generate_denizen_profile is from profile_image_generator
             logger.error("Fatal Error: Denizen components not available. Cannot run Denizen mode.")
             sys.exit(1)

        denizen_run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        denizen_output_dir = OUTPUT_DIR / f"run_denizen_{denizen_run_timestamp}"
        denizen_output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Using temporary output directory for Denizen run: {denizen_output_dir}")

        denizen_artifacts = {
             "markdown": None, "feature_image": None, "article_images": None,
             "audio": None, "video": None, "summary_file": None, "image_prompts_file": None
        }
        denizen_metadata = {}
        denizen_image_prompt_details = []

        try:
            denizen_data = generate_denizen_profile(denizen_output_dir) # This generates images and saves them in denizen_output_dir/images
            if not denizen_data:
                 raise RuntimeError("Failed to generate Dimensional Denizen profile data.")

            char_name = denizen_data['name']
            char_backstory = denizen_data['backstory']
            char_details = denizen_data['details']
            # Paths are already in denizen_data, relative to denizen_output_dir
            feature_image_path = denizen_data['profile_image_path']
            article_image_path = denizen_data['action_image_path']
            filename_base = denizen_data['filename_base'] # Use filename_base from denizen_data
            denizen_metadata['filename_base'] = filename_base
            denizen_metadata['location'] = char_details.get('theme', 'Unknown Dimension') # Add dimension to metadata (using 'theme' key from profile_image_generator)


            if feature_image_path: denizen_artifacts["feature_image"] = feature_image_path
            if article_image_path: denizen_artifacts["article_images"] = article_image_path

            if feature_image_path:
                denizen_image_prompt_details.append({
                    "image_type": "featured", "filename": feature_image_path.name,
                    "prompt": denizen_data.get("profile_image_prompt", "Prompt not available")
                })
            if article_image_path:
                 denizen_image_prompt_details.append({
                    "image_type": "article", "filename": article_image_path.name, "index": 0,
                    "prompt": denizen_data.get("action_image_prompt", "Prompt not available")
                })

            if denizen_image_prompt_details:
                prompts_json_path = denizen_output_dir / f"{filename_base}_image_prompts.json"
                try:
                    with open(prompts_json_path, 'w', encoding='utf-8') as f: json.dump(denizen_image_prompt_details, f, indent=2)
                    denizen_artifacts["image_prompts_file"] = prompts_json_path
                except Exception as e: logger.error(f"Failed to save denizen image prompts JSON: {e}")

            reporter_obj = Reporter(identifier='pixel_paradox')
            if not reporter_obj or not reporter_obj.reporter_data: raise RuntimeError("Could not init Pixel Paradox reporter.")

            category = "Dimensional Denizen"
            tags_list = ["The Ephergent", "Dimensional Denizen"] # Ensure it's a list
            selected_title = f"Daily Dimensional Denizen: {char_name}"
            summary = char_backstory # Use backstory as summary

            if "audio" not in steps_to_skip_list:
                logger.info("Generating Denizen Audio...")
                denizen_audio_text = prepare_denizen_text_for_tts(denizen_data, char_backstory)
                if denizen_audio_text:
                    audio_path = generate_article_audio(
                        reporter=reporter_obj, article_content=denizen_audio_text, title=selected_title,
                        output_dir=denizen_output_dir, filename_base=filename_base, speed=1.1
                    )
                    denizen_artifacts["audio"] = audio_path
                else: logger.warning("Failed to prepare Denizen text for audio.")
            else: logger.info("Skipping Denizen Audio.")

            if "video" not in steps_to_skip_list and denizen_artifacts.get("audio") and denizen_artifacts.get("feature_image"):
                 logger.info("Generating Denizen Video...")
                 denizen_video_output_dir = denizen_output_dir / "video"
                 denizen_video_output_dir.mkdir(exist_ok=True)
                 denizen_video_path = generate_youtube_video(
                     reporter=reporter_obj, title=selected_title, audio_path=denizen_artifacts["audio"],
                     featured_image_path=denizen_artifacts["feature_image"],
                     article_image_paths=[denizen_artifacts.get("article_images")] if denizen_artifacts.get("article_images") else [],
                     output_dir=denizen_video_output_dir, filename_base=filename_base
                 )
                 denizen_artifacts["video"] = denizen_video_path
            else: logger.info("Skipping Denizen Video.")

            denizen_youtube_url = None
            if "youtube" not in steps_to_skip_list and denizen_artifacts.get("video"):
                 logger.info("Uploading Denizen Video...")
                 denizen_video_id = upload_to_youtube(
                     video_file_path=denizen_artifacts["video"], title=selected_title, description=summary,
                     tags=tags_list, thumbnail_path=denizen_artifacts.get("feature_image")
                 )
                 if denizen_video_id:
                     denizen_youtube_url = f"https://www.youtube.com/watch?v={denizen_video_id}"
                     denizen_metadata['youtube_url'] = denizen_youtube_url
                 else: logger.warning("Denizen YouTube upload failed.")
            else: logger.info("Skipping Denizen YouTube Upload.")

            logger.info("Formatting Denizen Markdown...")
            pelican_markdown_content = format_denizen_pelican_markdown(
                char_name=char_name, char_backstory=char_backstory, char_details=char_details,
                feature_image_filename=denizen_artifacts["feature_image"].name if denizen_artifacts.get("feature_image") else None,
                article_image_filename=denizen_artifacts["article_images"].name if denizen_artifacts.get("article_images") else None,
                audio_filename=denizen_artifacts["audio"].name if denizen_artifacts.get("audio") else None,
                youtube_video_url=denizen_youtube_url, author=reporter_obj.name, category=category, tags=tags_list,
                publish_date=datetime.now() # Publish denizens immediately
            )
            markdown_temp_path = denizen_output_dir / f"{filename_base}.md"
            with open(markdown_temp_path, "w", encoding="utf-8") as f: f.write(pelican_markdown_content)
            denizen_artifacts["markdown"] = markdown_temp_path

            denizen_pelican_path = None
            if "export" not in steps_to_skip_list:
                 logger.info("Exporting Denizen to Pelican...")
                 denizen_media = {
                     "feature_image": denizen_artifacts.get("feature_image"),
                     "article_images": [denizen_artifacts.get("article_images")] if denizen_artifacts.get("article_images") else [],
                     "audio": denizen_artifacts.get("audio")
                 }
                 denizen_pelican_path = export_to_pelican(
                     markdown_content=pelican_markdown_content, markdown_filename_base=filename_base,
                     media_paths=denizen_media, is_denizen=True
                 )
                 if denizen_pelican_path: denizen_metadata['pelican_path'] = denizen_pelican_path
                 else: logger.error("Failed to export Denizen to Pelican.")
            else: logger.info("Skipping Denizen Pelican Export.")

            if "git" not in steps_to_skip_list and denizen_pelican_path:
                 logger.info("Publishing Denizen draft to Git...")
                 publish_to_git(f"Add denizen profile: {char_name}")
            else: logger.info("Skipping Denizen Git Publish.")

            if "archive" not in steps_to_skip_list and main_archiver and main_archiver.initialized:
                 logger.info("Archiving Denizen Artifacts...")
                 summary_path = denizen_output_dir / f"{filename_base}_summary.txt"
                 with open(summary_path, "w", encoding="utf-8") as f: f.write(summary)
                 denizen_artifacts["summary_file"] = summary_path
                 denizen_article_images_for_archive = [denizen_artifacts.get("article_images")] if denizen_artifacts.get("article_images") else []
                 denizen_artifacts_for_archive = denizen_artifacts.copy()
                 denizen_artifacts_for_archive["article_images"] = denizen_article_images_for_archive
                 archive_dir = main_archiver.archive_artifacts(filename_base, denizen_artifacts_for_archive)
                 if archive_dir: denizen_metadata['archive_path'] = archive_dir
                 else: logger.warning("Denizen archiving failed.")
            else: logger.info("Skipping Denizen Archiving.")

            if "mail" not in steps_to_skip_list and denizen_pelican_path:
                logger.info("Sending Denizen Email Notification...")
                try:
                    denizen_url = f"{BLOG_URL}/{PELICAN_DENIZENS_SUBDIR}/{filename_base}.html"
                    denizen_feature_image_url = f"{BLOG_URL}/{PELICAN_IMAGES_SUBDIR}/{denizen_artifacts['feature_image'].name}" if denizen_artifacts.get("feature_image") else f"{BLOG_URL}/theme/images/ephergent_logo.png"
                    email_vars = {
                        "newsletter_date": datetime.now().strftime("%B %d, %Y"), "article_title": selected_title,
                        "article_summary": summary, "article_url": denizen_url,
                        "article_feature_image_url": denizen_feature_image_url,
                    }
                    send_email(subject=f"Ephergent Denizen: {char_name}", template=os.getenv('MAILGUN_TEMPLATE', "daily.dimensional.dispatch"), variables=email_vars)
                except Exception as e: logger.error(f"Failed to send Denizen email: {e}", exc_info=True)
            else: logger.info("Skipping Denizen Email.")

            if "social" not in steps_to_skip_list and denizen_pelican_path:
                logger.info("Posting Denizen to Social Media...")
                try:
                    denizen_url = f"{BLOG_URL}/{PELICAN_DENIZENS_SUBDIR}/{filename_base}.html"
                    post_article_to_social_media(
                        title=selected_title, url=denizen_url, summary=summary,
                        image_path=denizen_artifacts.get("feature_image"), tags=tags_list
                    )
                except Exception as e: logger.error(f"Failed to post Denizen to social: {e}", exc_info=True)
            else: logger.info("Skipping Denizen Social Post.")
            logger.info("Denizen workflow finished successfully.")

        except Exception as e:
            logger.error(f"Denizen workflow failed: {e}", exc_info=True)
            sys.exit(1)
        finally:
            cleanup_output_directory(denizen_output_dir)

    elif args.auto_generate:
        # The generative workflow now handles saving to file and calling process_story_from_file
        run_generative_workflow(args.reporter, args.topic, steps_to_skip_list, main_archiver)

    else:
        # --- Process Stories from Input Directory (existing logic) ---
        logger.info(f"Loading stories from directory: {INPUT_STORIES_DIR_READY}")
        if not INPUT_STORIES_DIR_READY.is_dir():
            logger.warning(f"Input stories 'ready' directory not found: {INPUT_STORIES_DIR_READY}. No stories to process from files.")
            logger.info("Consider using --auto-generate to create new content or --denizen for a new denizen profile.")
            sys.exit(0)

        stories_to_process_files = sorted(list(INPUT_STORIES_DIR_READY.glob("*.json")))
        if not stories_to_process_files:
             logger.info("No story files found in 'ready' directory.")
             logger.info("Consider using --auto-generate to create new content or --denizen for a new denizen profile.")
             sys.exit(0)

        logger.info(f"Found {len(stories_to_process_files)} story files in 'ready' directory.")

        processed_count = 0
        fail_count = 0
        files_to_process = stories_to_process_files
        if args.limit > 0:
            logger.info(f"Limiting processing to the first {args.limit} stories.")
            files_to_process = stories_to_process_files[:args.limit]

        for story_file in files_to_process:
            try:
                match = re.match(r"(\d{3})_(\d{0,2})\.json", story_file.name) # Use updated pattern
                if not match:
                    logger.warning(f"Skipping file with unexpected name format: {story_file.name}")
                    continue
                month_num = int(match.group(1))
                # Handle missing/malformed week, default to 0
                try:
                    week_num = int(match.group(2)) if match.group(2) else 0
                except ValueError:
                    week_num = 0
                    logger.warning(f"Could not parse week from filename {story_file.name}. Treating as week 0.")

                with open(story_file, 'r', encoding='utf-8') as f: story_data_from_file = json.load(f)
                story_data_from_file['Month'], story_data_from_file['Week'] = month_num, week_num

                result = process_story_from_file(story_data_from_file, story_file, main_archiver, steps_to_skip_list)
                if result: processed_count += 1
                else: fail_count += 1
            except (json.JSONDecodeError, ValueError) as e:
                logger.error(f"Failed to parse or get date from {story_file.name}: {e}")
                fail_count += 1
            except Exception as e:
                logger.error(f"Failed to load/process story from {story_file.name}: {e}", exc_info=True)
                fail_count += 1

        logger.info(f"--- Story File Processing Complete ---")
        logger.info(f"Successfully processed: {processed_count}")
        logger.info(f"Failed: {fail_count}")
        if fail_count > 0: logger.warning("Some stories failed processing. Check logs.")

    logger.info("App execution finished.")
