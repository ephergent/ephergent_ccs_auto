#!/usr/bin/env python3
import os
import logging
import shutil
import re
from math import floor
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from typing import List, Optional, Dict, Any # Ensure Optional, Dict, Any are imported

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# --- Configuration ---
PELICAN_CONTENT_DIR = os.getenv('PELICAN_CONTENT_DIR')

# Define standard subdirectories within Pelican content
PELICAN_ARTICLES_SUBDIR = "articles"
PELICAN_IMAGES_SUBDIR = "images"
PELICAN_AUDIO_SUBDIR = "audio"
PELICAN_DENIZENS_SUBDIR = "denizens" # Example if separating Denizens
# PELICAN_CHAR_IMG_SUBDIR = "images/characters" # Removed as no longer used
# DEFAULT_AVATAR_PELICAN_PATH = "/images/ephergent_logo.png" # Removed as no longer used


def sanitize_filename(text: str, allow_unicode=False) -> str:
    """
    Sanitizes a string to be used as a filename slug.
    Adapted from Pelican's slugify function.
    """
    text = str(text)
    # Remove or replace problematic characters
    text = re.sub(r'[^\w\s-]', '', text, flags=re.U).strip().lower()
    # Replace whitespace and hyphens with a single hyphen
    text = re.sub(r'[-\s]+', '-', text)
    # Remove leading/trailing hyphens
    text = text.strip('-')
    # Handle potential empty strings after sanitization
    if not text:
        return f"sanitized-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    return text


# --- Formatting Functions ---
def format_story_markdown(
    story_data: Dict[str, Any],
    reporter_name: str, # Mapped from 'Filed by'
    category: str, # Determined based on reporter or context
    tags: List[str], # Determined based on reporter or context
    feature_image_filename: Optional[str] = None,
    article_image_filenames: Optional[List[str]] = None, # List of filenames
    audio_filename: Optional[str] = None,
    youtube_video_url: Optional[str] = None,
    publish_date: Optional[datetime] = None # Optional date for scheduling
    ) -> str:
    """
    Formats the article content and metadata from story JSON data into a Pelican Markdown string.

    Args:
        story_data (Dict[str, Any]): Dictionary containing story details from JSON
                                     (e.g., 'title', 'content', 'stardate', 'location').
        reporter_name (str): The name of the reporter.
        category (str): Article category.
        tags (List[str]): List of article tags.
        feature_image_filename: Filename of the feature image.
        article_image_filenames: List of filenames for in-article images.
        audio_filename: Filename of the associated audio file.
        youtube_video_url: Full URL to the YouTube video, if available.
        publish_date (Optional[datetime]): If provided, sets the 'Date' metadata. Otherwise, uses current time.

    Returns:
        The complete article content formatted as Pelican Markdown.
    """
    title = story_data.get('title', 'Untitled Ephergent Report')
    content = story_data.get('content', 'No content available.')
    stardate = story_data.get('stardate', '')
    location = story_data.get('location', 'Unknown Location')

    # Determine date: use publish_date if provided, else now
    article_date = publish_date if publish_date else datetime.now()
    date_str = article_date.strftime("%Y-%m-%d %H:%M:%S")

    slug = sanitize_filename(title)
    # Create a simple summary by taking the first paragraph or ~50 words
    first_paragraph = content.split('\n')[0]
    summary = ' '.join(first_paragraph.split()[:50])
    if len(first_paragraph.split()) > 50:
        summary += "..."
    summary = summary.replace('"', "'") # Avoid issues with quotes in metadata

    # --- Construct Metadata ---
    metadata_lines = [
        f"Title: {title}",
        f"Date: {date_str}",
        f"Category: {category}",
        f"Tags: {', '.join(tags)}",
        f"Slug: {slug}",
        f"Author: {reporter_name}",
        f"Summary: {summary}",
        f"Status: draft", # Default to draft
        # Add custom metadata from story_data if needed
        f"Stardate: {stardate}",
        f"Location: {location}",
    ]

    # Add featured characters metadata if provided
    featured_characters = story_data.get('featured_characters')
    if featured_characters and isinstance(featured_characters, list):
        metadata_lines.append(f"Featured_Characters: {', '.join(featured_characters)}")
        logger.info(f"Adding Featured_Characters metadata: {', '.join(featured_characters)}")

    # Add feature image metadata if provided
    if feature_image_filename:
        # Using a direct relative path from content root is often simplest
        # DO NOT TOUCH THIS CODE, IT IS CRUCIAL FOR PELICAN
        metadata_lines.append(f"Feature_image: {feature_image_filename}")
        metadata_lines.append(f"Feature_image_caption: {title}") # Optional caption

    # Add custom metadata for YouTube URL if provided
    if youtube_video_url:
        metadata_lines.append(f"YouTubeURL: {youtube_video_url}")
        logger.info(f"Adding YouTubeURL metadata: {youtube_video_url}")

    metadata = "\n".join(metadata_lines) + "\n\n" # Ensure blank line after metadata

    # --- Prepare Body Content ---
    raw_body_content = content.strip()
    processed_paragraphs_list = []
    paragraphs_from_content = raw_body_content.split('\n\n')

    for paragraph_text in paragraphs_from_content:
        # If a paragraph was intended as a blockquote (e.g., starts with '>'),
        # reformat it as regular text by removing the '>' markers from each line.
        if paragraph_text.strip().startswith(">"):
            cleaned_lines = []
            for line in paragraph_text.split('\n'):
                # Remove leading '>' and optional space. Also strip whitespace from the line.
                cleaned_line = re.sub(r"^\s*>\s?", "", line).strip()
                cleaned_lines.append(cleaned_line)
            # Join the cleaned lines back into a single paragraph string,
            # preserving internal newlines that were part of the original multi-line quote.
            processed_paragraphs_list.append("\n".join(cleaned_lines))
            logger.info("Reformatted a blockquote paragraph to plain text.")
        else:
            # Not a blockquote, add paragraph as is
            processed_paragraphs_list.append(paragraph_text)

    processed_body_content = '\n\n'.join(processed_paragraphs_list)


    # Insert article images HTML if provided
    if article_image_filenames:
        paragraphs_for_image_insertion = processed_body_content.split('\n\n') # Use the processed body
        num_paragraphs = len(paragraphs_for_image_insertion)
        num_images = len(article_image_filenames)
        insert_points = []

        # Calculate insertion points, distributing images somewhat evenly
        if num_paragraphs > 1 and num_images > 0:
            step = max(1, num_paragraphs // (num_images + 1))
            for i in range(num_images):
                insert_index = min(num_paragraphs -1, (i + 1) * step)
                # Avoid inserting at the very beginning if possible
                if insert_index == 0 and num_paragraphs > 1:
                    insert_index = 1
                insert_points.append(insert_index)
        elif num_images > 0: # If only one paragraph, append all images
             insert_points = [num_paragraphs] * num_images

        logger.info(f"Calculated insertion points {insert_points} for {num_images} images in {num_paragraphs} paragraphs.")

        # Insert images from last to first to avoid messing up indices
        inserted_count = 0
        for i, img_filename in reversed(list(enumerate(article_image_filenames))):
            if i < len(insert_points):
                insert_idx = insert_points[i]
                article_image_path_pelican = f"/{PELICAN_IMAGES_SUBDIR}/{img_filename}"
                image_html = (
                    f'<figure>\n'
                    f'  <img src="{article_image_path_pelican}" alt="Moment Captured by Luminara - Scene from {title}">\n'
                    f'  <figcaption>⁂ Moment Captured by Luminara ⁂</figcaption>\n'
                    f'</figure>'
                )
                if insert_idx >= len(paragraphs_for_image_insertion):
                    paragraphs_for_image_insertion.append(image_html) # Append if index is out of bounds
                else:
                    paragraphs_for_image_insertion.insert(insert_idx, image_html)
                inserted_count += 1
                logger.info(f"Inserted article image '{img_filename}' into markdown body at paragraph index {insert_idx}.")

        if inserted_count > 0:
            body_content_with_images = '\n\n'.join(paragraphs_for_image_insertion)
        else:
            body_content_with_images = processed_body_content # No images inserted
    else:
         body_content_with_images = processed_body_content # No images to insert

    # Final body content variable
    final_body_content = body_content_with_images

    # Add audio player HTML if provided (append at the end)
    if audio_filename:
        audio_html = (
            f'\n\n---\n\n' # Add separator
            f'<h4>Listen to this report:</h4>\n'
            f'<audio controls preload="metadata" style="width: 100%; max-width: 500px;">\n' # Style for responsiveness
            f'  <source src="/{PELICAN_AUDIO_SUBDIR}/{audio_filename}" type="audio/mpeg">\n' # Specify type
            f'  Your browser does not support the audio element.\n'
            f'</audio>\n'
            f'<figcaption style="font-size: 0.8em; color: grey;">⁂ Audio created by The Ephergent\'s dimensionally-aware AI [Note: voices may sound different in your dimension.] ⁂</figcaption>\n'
        )
        final_body_content += audio_html
        logger.info(f"Added audio player for '{audio_filename}' to markdown body.")

    # Add YouTube video link if provided (append after audio)
    if youtube_video_url:
        # Optionally add an embedded player (requires careful styling)
        video_id_match = re.search(r"v=([a-zA-Z0-9_-]+)", youtube_video_url)
        if video_id_match:
            video_id = video_id_match.group(1)
            embed_code = (f'<div style="position: relative; padding-bottom: 56.25%; height: 0; overflow: '
                          f'hidden; max-width: 100%; height: auto;"><iframe src="https://www.youtube.com/embed/{video_id}"'
                          f' frameborder="0" allowfullscreen style="position: absolute; top: 0; left: 0; '
                          f'width: 100%; height: 100%;"></iframe></div>\n'
                          f'<figcaption style="font-size: 0.8em; color: grey;">⁂ Video created by The Ephergent\'s '
                          f'dimensionally-aware AI ⁂</figcaption>\n')
            final_body_content += f"\n\n{embed_code}"
            logger.info(f"Added YouTube embed for '{youtube_video_url}' to markdown body.")
        else:
            logger.warning(f"Could not extract video ID from YouTube URL: {youtube_video_url}")
            # Fallback to simple link
            final_body_content += f'\n\n---\n\n[Watch the video report on YouTube]({youtube_video_url})\n'


    # --- Combine Metadata and Body ---
    full_markdown_content = metadata + final_body_content
    return full_markdown_content


def format_denizen_pelican_markdown(
    char_name: str,
    char_backstory: str,
    char_details: dict, # Contains details like home dimension, frequency etc. if available
    feature_image_filename: str | None, # Filename only (profile image)
    article_image_filename: str | None, # Filename only (action image)
    audio_filename: str | None,         # Filename only
    youtube_video_url: str | None,      # Add YouTube URL parameter
    author: str, # Should be Pixel Paradox
    category: str, # Should be Dimensional Denizen
    tags: list[str], # Should include Dimensional Denizen
    publish_date: Optional[datetime] = None # Optional date for scheduling
    ) -> str:
    """
    Formats the Dimensional Denizen profile into a Pelican Markdown string,
    matching the structure from the reference example and including audio/video links.

    Args:
        char_name: Character's name.
        char_backstory: Character's backstory.
        char_details: Dictionary of character details (may include specific fields).
        feature_image_filename: Filename of the feature image (profile).
        article_image_filename: Filename of the article image (action shot).
        audio_filename: Filename of the associated audio file.
        youtube_video_url: Full URL to the YouTube video, if available.
        author: Article author (Pixel Paradox).
        category: Article category (Dimensional Denizen).
        tags: List of article tags.
        publish_date (Optional[datetime]): If provided, sets the 'Date' metadata. Otherwise, uses current time.


    Returns:
        The complete article content formatted as Pelican Markdown.
    """
    # Determine date: use publish_date if provided, else now
    article_date = publish_date if publish_date else datetime.now()
    date_str = article_date.strftime("%Y-%m-%d %H:%M:%S")

    # Use a consistent slug format
    slug = sanitize_filename(f"Dimensional-Denizen-{char_name}")
    title = f"Dimensional Denizen: {char_name}" # Consistent title format
    summary = f"Profile of {char_name}, a denizen from dimension {char_details.get('home_dimension', 'Unknown')}."
    summary = summary.replace('"', "'") # Avoid issues with quotes in metadata

    # --- Construct Metadata ---
    metadata_lines = [
        f"Title: {title}",
        f"Date: {date_str}",
        f"Category: {category}", # Should be "Dimensional Denizen"
        f"Tags: {', '.join(tags)}", # Should include "Dimensional Denizen"
        f"Slug: {slug}",
        f"Author: {author}", # Should be "Pixel Paradox"
        f"Summary: {summary}",
        f"Status: draft", # Default to draft
    ]

    # Add feature image metadata if provided (used by some themes)
    if feature_image_filename:
        # DO NOT TOUCH THIS CODE, IT IS CRUCIAL FOR PELICAN
        metadata_lines.append(f"Feature_image: {feature_image_filename}")
        # Optional: Add caption if needed by theme
        metadata_lines.append(f"Feature_image_caption: {title}")

    # Add custom metadata for action image and YouTube URL if provided
    if article_image_filename:
        metadata_lines.append(f"ActionImage: /{PELICAN_IMAGES_SUBDIR}/{article_image_filename}")
    if youtube_video_url:
        metadata_lines.append(f"YouTubeURL: {youtube_video_url}")
        logger.info(f"Adding YouTubeURL metadata for Denizen: {youtube_video_url}")

    metadata = "\n".join(metadata_lines) + "\n\n" # Ensure blank line after metadata

    # --- Prepare Body Content (matching ref_code example) ---
    body_parts = []

    body_parts.append("## TODAY'S DIMENSIONAL DENIZEN")

    # Article Image (Action Shot)
    if article_image_filename:
        article_image_path_pelican = f"/{PELICAN_IMAGES_SUBDIR}/{article_image_filename}"
        body_parts.append(
            f'<figure>\n'
            f'  <img src="{article_image_path_pelican}" alt="{title} - Action Shot">\n'
            f'  <figcaption>⁂ Moment Captured by Luminara ⁂</figcaption>\n'
            f'</figure>'
        )
    else:
        body_parts.append("*Action image generation failed or was skipped.*")

    # Character Name Heading
    body_parts.append(f"### {char_name.upper()}")

    # Add details if available in the char_details dict (adapt keys as needed)
    details_md = "**DETAILS:**\n\n"
    details_list = []
    # Use details from profile_image_generator.py output
    if 'appearance' in char_details: details_list.append(f"* **Appearance:** {char_details['appearance']}")
    if 'char_type' in char_details: details_list.append(f"* **Type:** {char_details['char_type']}")
    if 'gender' in char_details: details_list.append(f"* **Gender:** {char_details['gender']}")
    if 'hair' in char_details: details_list.append(f"* **Hair:** {char_details['hair']}")
    if 'eyes' in char_details: details_list.append(f"* **Eyes:** {char_details['eyes']}")
    if 'clothing' in char_details: details_list.append(f"* **Clothing:** {char_details['clothing']}")
    if 'accessory' in char_details: details_list.append(f"* **Accessory:** {char_details['accessory']}")
    if 'expression' in char_details: details_list.append(f"* **Expression:** {char_details['expression']}")
    if 'theme' in char_details: details_list.append(f"* **Theme:** {char_details['theme']}")
    # Add others if they exist in the dict
    for key, value in char_details.items():
        # Avoid duplicating already added keys
        if key not in ['name', 'appearance', 'char_type', 'gender', 'hair', 'eyes', 'clothing', 'accessory', 'expression', 'theme']:
            # Simple formatting for unknown keys
            formatted_key = key.replace('_', ' ').title()
            details_list.append(f"* **{formatted_key}:** {value}")


    if details_list:
        body_parts.append(details_md + "\n".join(details_list))

    body_parts.append("---")

    # Backstory
    body_parts.append("### BACKSTORY\n\n" + char_backstory)

    # Add other sections if they exist in char_details (like PERSONAL LIFE, CONTROVERSY)
    # These keys might not be generated by the current profile_image_generator
    if 'personal_life' in char_details:
        body_parts.append("\n---\n\n### PERSONAL LIFE\n\n" + char_details['personal_life'])
    if 'controversy' in char_details:
        body_parts.append("\n---\n\n### CONTROVERSY\n\n" + char_details['controversy'])
    if 'probability_of_encounter' in char_details:
        body_parts.append("\n---\n\n### PROBABILITY OF ENCOUNTER\n\n" + char_details['probability_of_encounter'])


    # Audio Player
    if audio_filename:
        body_parts.append("\n---") # Separator before audio
        audio_html = (
            f'<h4>Listen to the Profile:</h4>\n'
            f'<audio controls preload="metadata" style="width: 100%; max-width: 500px;">\n'
            f'  <source src="/{PELICAN_AUDIO_SUBDIR}/{audio_filename}" type="audio/mpeg">\n'
            f'  Your browser does not support the audio element.\n'
            f'</audio>\n'
            f'<figcaption style="font-size: 0.8em; color: grey;">⁂ Audio created by The Ephergent\'s dimensionally-aware AI ⁂</figcaption>'
        )
        body_parts.append(audio_html)
        logger.info(f"Added audio player for Denizen '{audio_filename}' to markdown body.")

    # YouTube Video Link
    if youtube_video_url:
        body_parts.append("\n---")
        video_id_match = re.search(r"v=([a-zA-Z0-9_-]+)", youtube_video_url)
        if video_id_match:
            video_id = video_id_match.group(1)
            embed_code = (f'<div style="position: relative; padding-bottom: 56.25%; height: 0; overflow: '
                          f'hidden; max-width: 100%; height: auto;"><iframe src="https://www.youtube.com/embed/{video_id}" '
                          f'frameborder="0" allowfullscreen style="position: absolute; top: 0; left: 0; width: 100%; '
                          f'height: 100%;"></iframe></div>\n' # Removed extra /n
                          f'<figcaption style="font-size: 0.8em; color: grey;">⁂ Video created by The Ephergent\'s '
                          f'dimensionally-aware AI ⁂</figcaption>\n')
            body_parts.append(f"\n\n{embed_code}") # Add spacing before embed
            logger.info(f"Added YouTube embed for Denizen '{youtube_video_url}' to markdown body.")
        else:
            logger.warning(f"Could not extract video ID from YouTube URL: {youtube_video_url}")
            # Fallback to simple link
            body_parts.append(f'\n\n---\n\n[Watch the video profile on YouTube]({youtube_video_url})\n')


    body_content = "\n\n".join(body_parts) # Join sections with double newlines

    # --- Combine Metadata and Body ---
    full_markdown_content = metadata + body_content.strip()
    logger.info(f"Formatted Denizen profile markdown for: {char_name}")
    return full_markdown_content


def export_to_pelican(
    markdown_content: str,
    markdown_filename_base: str, # e.g., "YYYYMMDD_sanitized_title"
    media_paths: Dict[str, Path | List[Path] | None], # Use dict for various media types
    is_denizen: bool = False # Flag to potentially use different subdirs
    ) -> Path | None:
    """
    Saves the formatted markdown and copies associated media files to the Pelican content directory.

    Args:
        markdown_content (str): The fully formatted Pelican markdown string.
        markdown_filename_base (str): Base name for the markdown file (without extension).
        media_paths (Dict[str, Path | List[Path] | None]): Dictionary mapping media types
            (e.g., 'feature_image', 'article_images', 'audio') to their source Path objects
            or list of Path objects. Values can be None if not generated.
        is_denizen (bool): If True, potentially export to a different subdirectory.

    Returns:
        Path | None: The path to the created Markdown file in the Pelican content directory,
                     or None if the export failed.
    """
    if not PELICAN_CONTENT_DIR:
        logger.error("PELICAN_CONTENT_DIR environment variable not set. Cannot export.")
        return None

    pelican_content_path = Path(PELICAN_CONTENT_DIR)
    if not pelican_content_path.is_dir():
        logger.error(f"PELICAN_CONTENT_DIR '{PELICAN_CONTENT_DIR}' does not exist or is not a directory.")
        return None

    # Determine target subdirectories
    if is_denizen:
         articles_target_dir = pelican_content_path / PELICAN_DENIZENS_SUBDIR
         logger.info(f"Exporting Denizen profile to: {articles_target_dir}")
    else:
         articles_target_dir = pelican_content_path / PELICAN_ARTICLES_SUBDIR
         logger.info(f"Exporting article to: {articles_target_dir}")

    target_images_dir = pelican_content_path / PELICAN_IMAGES_SUBDIR
    target_audio_dir = pelican_content_path / PELICAN_AUDIO_SUBDIR

    # Create target directories if they don't exist
    try:
        articles_target_dir.mkdir(parents=True, exist_ok=True)
        target_images_dir.mkdir(parents=True, exist_ok=True)
        target_audio_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.error(f"Error creating target directories in '{pelican_content_path}': {e}")
        return None

    # --- Save Markdown File ---
    markdown_filename = f"{markdown_filename_base}.md"
    target_markdown_path = articles_target_dir / markdown_filename
    try:
        with open(target_markdown_path, "w", encoding="utf-8") as f:
            f.write(markdown_content)
        logger.info(f"Successfully saved Markdown to: {target_markdown_path}")
    except IOError as e:
        logger.error(f"Failed to save Markdown file to {target_markdown_path}: {e}")
        return None # Fail early if markdown can't be saved

    # --- Copy Media Files ---
    files_copied_successfully = True

    def copy_file(source_path: Path, target_dir: Path, file_type: str) -> bool:
        """Helper function to copy a single file."""
        if source_path and source_path.is_file():
            target_path = target_dir / source_path.name
            try:
                shutil.copy2(source_path, target_path)
                logger.info(f"Copied {file_type} to: {target_path}")
                return True
            except Exception as e:
                logger.error(f"Failed to copy {file_type} from {source_path} to {target_path}: {e}")
                return False
        elif source_path:
            logger.warning(f"{file_type.capitalize()} source path not found or is not a file: {source_path}")
            return False
        return True # Return True if source_path was None (nothing to copy)

    # Copy Feature Image
    feature_image_path = media_paths.get("feature_image")
    if isinstance(feature_image_path, Path):
        if not copy_file(feature_image_path, target_images_dir, "feature image"):
            files_copied_successfully = False
    elif feature_image_path:
        logger.warning(f"Invalid type for feature_image path: {type(feature_image_path)}")

    # Copy Article Images (handle list)
    article_image_paths = media_paths.get("article_images")
    if isinstance(article_image_paths, list):
        for img_path in article_image_paths:
            if isinstance(img_path, Path):
                if not copy_file(img_path, target_images_dir, "article image"):
                    files_copied_successfully = False
            elif img_path:
                 logger.warning(f"Invalid type for item in article_images list: {type(img_path)}")
    elif article_image_paths:
         logger.warning(f"Invalid type for article_images: {type(article_image_paths)}")


    # Copy Audio File
    audio_path = media_paths.get("audio")
    if isinstance(audio_path, Path):
        if not copy_file(audio_path, target_audio_dir, "audio file"):
            files_copied_successfully = False
    elif audio_path:
        logger.warning(f"Invalid type for audio path: {type(audio_path)}")

    # Note: Video files are not handled by this exporter currently.
    # They are archived by the archiver utility.

    if not files_copied_successfully:
        logger.warning("One or more media files failed to copy. Markdown file was still created.")
        # Decide if this constitutes failure - returning the markdown path anyway for now
        return target_markdown_path

    logger.info("Pelican export completed successfully.")
    return target_markdown_path


# Example usage when run directly
if __name__ == "__main__":
    print("Testing Pelican Exporter...")

    if not PELICAN_CONTENT_DIR:
        print("PELICAN_CONTENT_DIR environment variable not set. Skipping test.")
    else:
        print(f"Using Pelican Content Directory: {PELICAN_CONTENT_DIR}")

        # Create dummy artifacts for testing
        temp_output_dir = Path("./output_temp_test")
        temp_output_dir.mkdir(exist_ok=True)
        temp_images_dir = temp_output_dir / "images"
        temp_images_dir.mkdir(exist_ok=True)
        temp_audio_dir = temp_output_dir / "audio" # Audio now saved in base temp dir
        temp_audio_dir.mkdir(exist_ok=True)

        dummy_feature_img = temp_images_dir / "test_feature_image.png"
        dummy_article_img1 = temp_images_dir / "test_article_image1.jpg"
        dummy_article_img2 = temp_images_dir / "test_article_image2.gif"
        dummy_audio = temp_audio_dir / "test_audio_combined.mp3" # Adjusted path

        try:
            # Create dummy files
            dummy_feature_img.touch()
            dummy_article_img1.touch()
            dummy_article_img2.touch()
            dummy_audio.touch()
            print("Created dummy artifact files.")

            # Dummy story data (simulating JSON input)
            test_story_data = {
                "title": "Pelican Export Test: The Quantum Quandary",
                "content": (
                    "This is the first paragraph of the test article.\n\n"
                    "> This is a blockquote that should be converted to normal text.\n"
                    "> It has multiple lines.\n\n"
                    "This is the second paragraph. We expect an image to be inserted after this one.\n\n"
                    "This is the third paragraph, appearing after the image. It discusses the implications of testing."
                ),
                "stardate": "81.2 Glitch Standard",
                "location": "Test Dimension - Debug Sector"
            }
            test_reporter_name = "Test Reporter"
            test_category = "Tests"
            test_tags = ["testing", "pelican", "export"]
            test_youtube_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ" # Example URL
            test_filename_base = f"{datetime.now().strftime('%Y%m%d')}_{sanitize_filename(test_story_data['title'])}"

            # 1. Format Markdown using new function
            print("\nFormatting Story Markdown...")
            formatted_md = format_story_markdown(
                story_data=test_story_data,
                reporter_name=test_reporter_name,
                category=test_category,
                tags=test_tags,
                feature_image_filename=dummy_feature_img.name,
                article_image_filenames=[dummy_article_img1.name, dummy_article_img2.name], # Pass list
                audio_filename=dummy_audio.name,
                youtube_video_url=test_youtube_url # Pass the test URL
            )
            print("--- Formatted Markdown Preview ---")
            # Expected output for the blockquote part:
            # "This is a blockquote that should be converted to normal text.\nIt has multiple lines."
            print(formatted_md[:800] + "\n...") # Show more preview
            print("--- End Preview ---")

            # 2. Export to Pelican
            print("\nExporting to Pelican...")
            media_to_export = {
                "feature_image": dummy_feature_img,
                "article_images": [dummy_article_img1, dummy_article_img2], # Pass list
                "audio": dummy_audio
            }
            exported_path = export_to_pelican(
                markdown_content=formatted_md,
                markdown_filename_base=test_filename_base,
                media_paths=media_to_export,
                is_denizen=False
            )

            if exported_path:
                print(f"\nExport successful! Markdown file created at: {exported_path}")
                print("Check your Pelican content directory for the new files:")
                print(f"- {PELICAN_CONTENT_DIR}/{PELICAN_ARTICLES_SUBDIR}/{test_filename_base}.md")
                print(f"- {PELICAN_CONTENT_DIR}/{PELICAN_IMAGES_SUBDIR}/{dummy_feature_img.name}")
                print(f"- {PELICAN_CONTENT_DIR}/{PELICAN_IMAGES_SUBDIR}/{dummy_article_img1.name}")
                print(f"- {PELICAN_CONTENT_DIR}/{PELICAN_IMAGES_SUBDIR}/{dummy_article_img2.name}")
                print(f"- {PELICAN_CONTENT_DIR}/{PELICAN_AUDIO_SUBDIR}/{dummy_audio.name}")
            else:
                print("\nExport failed.")

        except Exception as e:
            print(f"\nAn error occurred during the test: {e}")
        finally:
            # Clean up dummy files/dir
            if temp_output_dir.exists():
                print("\nCleaning up temporary test directory...")
                shutil.rmtree(temp_output_dir)
                print("Cleanup complete.")
