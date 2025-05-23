#!/usr/bin/env python3
"""
YouTube Video Generator Utility

Generates an MP4 video from audio, images, and reporter information.
Adapted from ref_code/youtube_video_generator.py for integration into the main app.
"""

import logging
from pathlib import Path
import os
import random
import math
from typing import Tuple, Optional, List, Sequence, Any

# Third-party libraries
try:
    from moviepy import ( # Use moviepy.editor for easier access
        AudioFileClip, ImageClip, VideoClip, concatenate_videoclips, TextClip,
        CompositeVideoClip, ColorClip
    )
    from moviepy.video import fx as vfx # Import effects module with an alias
    import numpy as np
    from PIL import Image, ImageDraw, ImageOps, UnidentifiedImageError
    MOVIEPY_AVAILABLE = True
except ImportError:
    MOVIEPY_AVAILABLE = False
    # Define dummy classes/functions if moviepy is not installed
    # This allows the rest of the app to import this module without crashing
    # if the video generation dependencies are missing.
    class DummyClip:
        def __init__(self, *args, **kwargs): pass
        duration = 0
        size = (0, 0)
        with_duration = with_fps = with_position = with_start = with_audio = with_mask = lambda self, *args, **kwargs: self
        close = write_videofile = lambda self, *args, **kwargs: None

    AudioFileClip = ImageClip = VideoClip = TextClip = CompositeVideoClip = ColorClip = DummyClip
    concatenate_videoclips = lambda clips, **kwargs: clips[0] if clips else DummyClip()
    vfx = None # Effects module unavailable
    np = None
    Image = ImageDraw = ImageOps = UnidentifiedImageError = None


# --- Project Imports ---
# Assuming Reporter class is available via utils.reporter
try:
    from utils.reporter import Reporter
except ImportError:
    # Define a dummy Reporter class if it cannot be imported
    # This might happen if running this script standalone without the full project structure
    class Reporter:
        def __init__(self, identifier: str):
            self.id = identifier
            self.name = "Unknown Reporter"
            self.stable_diffusion_prompt = "" # Add property used
        # Add any other methods/properties accessed by this module if necessary

# --- Constants ---
logger = logging.getLogger(__name__)

# Define base asset paths relative to the project root
PROJECT_ROOT = Path(__file__).parent.parent
ASSETS_DIR = PROJECT_ROOT / 'assets'
DEFAULT_LOGO_IMAGE_PATH = ASSETS_DIR  / 'images'  / 'ephergent_logo.png'
DEFAULT_FONT_PATH = ASSETS_DIR / 'fonts' / 'Montserrat-Black.ttf' # Example font
REPORTER_IMAGES_DIR = ASSETS_DIR / 'images' / 'reporters'

DEFAULT_LOGO_DURATION = 2.0 # seconds
DEFAULT_FEATURED_DURATION = 4.0  # seconds
DEFAULT_ARTICLE_IMAGE_DURATION = 5.0 # seconds per image (increased slightly)
DEFAULT_RESOLUTION = (1280, 720) # Width, Height tuple
DEFAULT_ANIMATION = 'slide'
VIDEO_FPS = 24 # Standard video FPS (increased from 12)
ANIMATION_TYPES = ['slide'] # Only 'slide' is currently implemented
MAX_FEATURED_DURATION_RATIO = 0.15 # Max 15% of total audio for featured image
REPORTER_SIZE_RATIO = 0.20 # Reporter circle diameter relative to video height (slightly smaller)
REPORTER_MARGIN_RATIO = 0.03 # Margin for reporter relative to video dims (slightly smaller)
TITLE_FONT_SIZE_RATIO = 0.04 # Font size relative to video height (slightly smaller)
TITLE_MARGIN_RATIO = 0.04 # Margin for title relative to video dims
GRADIENT_HEIGHT_RATIO = 0.15 # Height of gradient overlay at bottom (increased for better text visibility)

# --- Utility Functions ---

def get_resolution(resolution_input: Any, default_resolution: Tuple[int, int] = DEFAULT_RESOLUTION) -> Tuple[int, int]:
    """Convert 'WxH' string or tuple to validated (width, height) tuple."""
    default_width, default_height = default_resolution

    # If resolution_input is already a tuple, validate it
    if isinstance(resolution_input, tuple) and len(resolution_input) == 2:
        width, height = resolution_input
        if isinstance(width, int) and isinstance(height, int) and width > 0 and height > 0:
            return width, height
        else:
            logger.warning(f"Invalid resolution tuple {resolution_input}. Using default {default_width}x{default_height}.")
            return default_width, default_height

    # Otherwise process as string
    if isinstance(resolution_input, str):
        try:
            width, height = map(int, resolution_input.lower().split('x'))
            if width <= 0 or height <= 0:
                raise ValueError("Width and height must be positive.")
            return width, height
        except (ValueError, AttributeError) as e:
            logger.warning(
                f"Invalid resolution format '{resolution_input}'. {e}. Using default {default_width}x{default_height}.")
            return default_width, default_height
    else:
        logger.warning(f"Invalid resolution input type '{type(resolution_input)}'. Using default {default_width}x{default_height}.")
        return default_width, default_height


def _create_circular_mask(size: Tuple[int, int]) -> Optional[Any]:
    """Create a circular mask (alpha channel) of the given size."""
    if not Image or not ImageDraw:
        logger.warning("PIL/Pillow is not available, skipping circular mask creation.")
        return None
    mask = Image.new('L', size, 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((0, 0) + size, fill=255)
    return mask

def _create_gradient_mask(width: int, height: int, gradient_height: int) -> Optional[Any]:
    """Create a gradient mask that fades from transparent to black at the bottom."""
    if not np or not Image:
        logger.warning("NumPy or PIL/Pillow not available, skipping gradient mask creation.")
        return None
    try:
        # Create an array for the alpha channel directly
        alpha_gradient = np.zeros((height, width), dtype=np.uint8)
        start_y = height - gradient_height

        # Create linear gradient from 0 to 255 over the gradient height
        gradient_values = np.linspace(0, 255, gradient_height, dtype=np.uint8)

        # Apply gradient to the alpha channel in the specified region
        for i, alpha_val in enumerate(gradient_values):
            y_pos = start_y + i
            if 0 <= y_pos < height:
                alpha_gradient[y_pos, :] = alpha_val

        # Convert alpha channel numpy array to PIL Image (mode 'L')
        gradient_mask_img = Image.fromarray(alpha_gradient, mode='L')
        return gradient_mask_img
    except Exception as e:
        logger.error(f"Error creating gradient mask: {e}", exc_info=True)
        return None


# --- Animation Effects ---

def create_slide_effect(image_clip: ImageClip, duration: float, width: int, height: int) -> ImageClip:
    """
    Placeholder for slide effect. Currently returns the clip unmodified
    to address potential centering issues caused by animation offsets/zoom.
    TODO: Re-implement slide/zoom effect carefully if desired, ensuring it doesn't affect perceived centering.
    """
    logger.debug("Slide effect currently disabled. Returning static clip.")
    # Return the clip without applying any transformation
    return image_clip


# --- Core Video Creation Logic ---

def _load_and_prepare_image(image_path: Path, size: Tuple[int, int]) -> Optional[Any]:
    """Load an image, resize it, and return as numpy array."""
    if not MOVIEPY_AVAILABLE or not Image or not np: return None # Handle missing dependency
    try:
        img = Image.open(image_path)
        # Handle RGBA images: create white background and paste RGBA image onto it
        if img.mode == 'RGBA':
            bg = Image.new('RGB', img.size, (255, 255, 255))
            bg.paste(img, mask=img.split()[3]) # Paste using alpha channel as mask
            img = bg
        else:
            img = img.convert("RGB")  # Ensure 3 channels for other modes

        # Use LANCZOS for resizing
        img_resized = ImageOps.fit(img, size, Image.Resampling.LANCZOS)
        return np.array(img_resized)
    except FileNotFoundError:
        logger.error(f"Error: Image file not found at {image_path}")
        return None
    except UnidentifiedImageError:
        logger.error(f"Error: Cannot identify image file format at {image_path}")
        return None
    except Exception as e:
        logger.error(f"Error processing image {image_path}: {e}", exc_info=True)
        return None

def _apply_animation(
    clip: ImageClip, animation_type: str, duration: float, width: int, height: int
) -> ImageClip:
    """Apply the selected animation effect to the image clip."""
    if not MOVIEPY_AVAILABLE: return clip # Return original if deps missing

    logger.info(f"Applying '{animation_type}' animation...")
    if animation_type == 'slide':
        return create_slide_effect(clip, duration, width, height)
    else:
        logger.warning(f"Unsupported animation type '{animation_type}'. Using default 'slide'.")
        return create_slide_effect(clip, duration, width, height) # Fallback

def generate_youtube_video(
    reporter: Reporter,
    title: str,
    audio_path: Path,
    featured_image_path: Path,
    article_image_paths: Sequence[Path],  # Use Sequence for flexible input
    output_dir: Path,
    filename_base: str,
    resolution_input: Any = DEFAULT_RESOLUTION,  # Resolution as string or tuple
    animation: str = DEFAULT_ANIMATION,
    font_path: Path = DEFAULT_FONT_PATH,
    logo_path: Path = DEFAULT_LOGO_IMAGE_PATH,
    ) -> Optional[Path]:
    """
    Generates a YouTube-ready MP4 video.

    Args:
        reporter (Reporter): The Reporter object for avatar info.
        title (str): The video title overlay text.
        audio_path (Path): Path to the input MP3 audio file.
        featured_image_path (Path): Path to the featured image (used for intro and thumbnail).
        article_image_paths (Sequence[Path]): List of paths to article images.
        output_dir (Path): Directory to save the generated video file.
        filename_base (str): Base name for the output video file (e.g., "my_article_video").
        resolution_input (Any): Video resolution ('WxH' string or (width, height) tuple).
        animation (str): Animation type for the article image section.
        font_path (Path): Path to the TTF/OTF font file for the title overlay.
        logo_path (Path): Path to the logo image file.

    Returns:
        Optional[Path]: Path to the generated MP4 video file, or None if generation failed.
    """
    if not MOVIEPY_AVAILABLE:
        logger.error("MoviePy or its dependencies (Pillow, NumPy) not installed. Cannot generate video.")
        return None

    output_path = output_dir / f"{filename_base}_video.mp4"
    width, height = get_resolution(resolution_input)
    size = (width, height)

    # --- 1. Input Validation and Setup ---
    logger.info("Starting video generation process...")
    logger.info(f"Output path: {output_path}")
    logger.info(f"Resolution: {width}x{height}")

    if not audio_path or not audio_path.exists():
        logger.error(f"Audio file not found or not provided: {audio_path}")
        return None
    if not logo_path.exists():
        logger.warning(f"Logo image not found at {logo_path}, skipping logo.")
        logo_path = None # Set to None if not found
    if not featured_image_path or not featured_image_path.exists():
        logger.error(f"Featured image file not found or not provided: {featured_image_path}")
        return None

    # Check article images
    valid_article_image_paths = []
    if article_image_paths: # Check if the sequence itself is not None or empty
        for img_path in article_image_paths:
            if img_path and img_path.exists():
                valid_article_image_paths.append(img_path)
            elif img_path:
                logger.warning(f"Article image file not found: {img_path}")
            # else: skip None entries silently
    else:
        logger.warning("No article image paths provided.")


    if not valid_article_image_paths:
        logger.warning("No valid article images found. Video will only use featured image.")
        # If no article images, the main section will just be the featured image

    if not font_path.exists():
        logger.warning(f"Font file not found at {font_path}. TextClip might use default font.")
        # Moviepy might still work with a default font if ImageMagick is configured

    # Determine reporter image path (assuming convention: reporter_id.png)
    reporter_image_name = f"{reporter.id}.png" # Or fetch from reporter object if available
    reporter_image_path = REPORTER_IMAGES_DIR / reporter_image_name
    if not reporter_image_path.exists():
        logger.warning(f"Reporter image not found at {reporter_image_path}. Skipping reporter overlay.")
        reporter_image_path = None # Set to None if not found

    output_dir.mkdir(parents=True, exist_ok=True)

    # --- 2. Load Assets ---
    logger.info("Loading assets...")
    audio_clip = None # Initialize
    clip_instances = [] # Keep track of instances to close later

    try:
        audio_clip = AudioFileClip(str(audio_path))
        total_duration = audio_clip.duration
        if total_duration <= 0:
            logger.error(f"Audio file has zero or negative duration: {audio_path}")
            return None
        logger.info(f"Audio duration: {total_duration:.2f} seconds")
        clip_instances.append(audio_clip) # Add to list for closing
    except Exception as e:
        logger.error(f"Error loading audio file {audio_path}: {e}", exc_info=True)
        return None

    # Load images
    logo_img_np = _load_and_prepare_image(logo_path, size) if logo_path else None
    featured_img_np = _load_and_prepare_image(featured_image_path, size)
    if featured_img_np is None:
        logger.error("Failed to load or prepare featured image.")
        # Close already opened clips before returning
        for clip in clip_instances: clip.close()
        return None

    # Load article images
    article_images_np = []
    for img_path in valid_article_image_paths:
        img_np = _load_and_prepare_image(img_path, size)
        if img_np is None:
            logger.warning(f"Failed to load or prepare article image: {img_path}")
        else:
            article_images_np.append(img_np)

    if not article_images_np:
        logger.warning("No valid article images were loaded. Using featured image for main section.")
        # Use featured image if no article images are available
        article_images_np.append(featured_img_np)


    reporter_img_pil = None
    if reporter_image_path:
        try:
            reporter_img_pil = Image.open(reporter_image_path).convert("RGBA")
        except Exception as e:
            logger.warning(f"Could not load reporter image {reporter_image_path}: {e}. Skipping overlay.")
            reporter_img_pil = None

    # --- 3. Calculate Durations ---
    logo_duration = DEFAULT_LOGO_DURATION if logo_img_np is not None else 0
    logo_duration = min(logo_duration, total_duration) # Ensure logo duration isn't longer than audio

    # Calculate max featured duration based on remaining time
    remaining_after_logo = total_duration - logo_duration
    max_feat_dur = remaining_after_logo * MAX_FEATURED_DURATION_RATIO

    # Use default featured duration, but cap it
    featured_duration = min(DEFAULT_FEATURED_DURATION, max_feat_dur, remaining_after_logo)
    featured_duration = max(0, featured_duration) # Ensure non-negative

    article_section_duration = total_duration - logo_duration - featured_duration
    article_section_duration = max(0, article_section_duration) # Ensure non-negative

    # Adjust durations slightly if rounding causes issues, ensure sum matches total_duration
    final_total_calculated = logo_duration + featured_duration + article_section_duration
    duration_diff = total_duration - final_total_calculated
    if abs(duration_diff) > 0.01: # Allow small tolerance
        logger.warning(f"Adjusting calculated durations (diff: {duration_diff:.3f}s) to match audio duration.")
        # Add difference to the longest section (usually article section)
        article_section_duration += duration_diff
        article_section_duration = max(0, article_section_duration) # Ensure non-negative again

    logger.info(f"Calculated Durations: Logo={logo_duration:.2f}s, Featured={featured_duration:.2f}s, Article={article_section_duration:.2f}s (Total: {logo_duration + featured_duration + article_section_duration:.2f}s)")

    if article_section_duration < 0.1 and len(article_images_np) > 0:
        logger.warning("Article section duration is very short (<0.1s).")
        # If article section is too short, potentially merge into featured duration?
        # For now, we allow it, but overlays might look weird.

    # --- 4. Create Base Video Clips ---
    logger.info("Creating base video clips...")
    all_clips = []

    # Create logo clip
    if logo_img_np is not None and logo_duration > 0.01:
        logo_clip = ImageClip(logo_img_np).with_duration(logo_duration).with_fps(VIDEO_FPS)
        all_clips.append(logo_clip)
        clip_instances.append(logo_clip)

    # Create featured clip
    if featured_img_np is not None and featured_duration > 0.01:
        # Simple static featured image clip
        featured_clip = ImageClip(featured_img_np).with_duration(featured_duration).with_fps(VIDEO_FPS)
        all_clips.append(featured_clip)
        clip_instances.append(featured_clip)

    # Create article clips (looping)
    article_clips = []
    if article_section_duration > 0.01 and article_images_np:
        logger.info(f"Creating looping article image clips (each ~{DEFAULT_ARTICLE_IMAGE_DURATION}s)...")
        remaining_article_duration = article_section_duration
        img_idx = 0
        while remaining_article_duration > 0.01 and article_images_np:
            # Determine duration for this specific clip
            current_clip_duration = min(DEFAULT_ARTICLE_IMAGE_DURATION, remaining_article_duration)

            img_np = article_images_np[img_idx % len(article_images_np)]
            article_clip_base = ImageClip(img_np).with_duration(current_clip_duration)

            # Apply animation
            article_clip_animated = _apply_animation(
                article_clip_base, animation, current_clip_duration, width, height
            )
            # Ensure FPS is set after animation
            article_clip_animated = article_clip_animated.with_fps(VIDEO_FPS)
            article_clips.append(article_clip_animated)
            clip_instances.append(article_clip_animated) # Track instance

            remaining_article_duration -= current_clip_duration
            img_idx += 1
            if img_idx > 100:  # Safety break for potential infinite loop
                logger.warning("Exceeded maximum article clip iterations (100).")
                break

        if article_clips:
            # Concatenate the animated article clips
            concatenated_articles = concatenate_videoclips(article_clips, method="compose")
            # Ensure the concatenated clip has the exact required duration
            concatenated_articles = concatenated_articles.with_duration(article_section_duration)
            all_clips.append(concatenated_articles)
            clip_instances.append(concatenated_articles)
        else:
            logger.warning("No article clips were generated.")
    elif not article_images_np:
        logger.warning("No article images available for the main section.")


    # --- 5. Concatenate Base Clips ---
    if not all_clips:
        logger.error("No video clips could be generated. Check durations and inputs.")
        # Close already opened clips before returning
        for clip in clip_instances: clip.close()
        return None

    logger.info("Combining base clips...")
    try:
        base_video = concatenate_videoclips(all_clips, method="compose")
        # Ensure final base video duration matches the calculated total precisely
        final_base_duration = logo_duration + featured_duration + article_section_duration
        base_video = base_video.with_duration(final_base_duration)
        clip_instances.append(base_video)
    except Exception as e:
        logger.error(f"Error concatenating base video clips: {e}", exc_info=True)
        for clip in clip_instances: clip.close()
        return None


    # --- 6. Create Overlays (Reporter, Title, Gradient) ---
    overlays = [base_video] # Start with the base video
    article_start_time = logo_duration + featured_duration

    gradient_clip = None
    reporter_clip = None
    title_clip = None

    # Create gradient overlay for bottom of screen (during article section)
    if article_section_duration > 0.01:
        logger.info("Creating gradient overlay...")
        gradient_height_px = int(height * GRADIENT_HEIGHT_RATIO)
        gradient_mask_pil = _create_gradient_mask(width, height, gradient_height_px)
        if gradient_mask_pil is not None and np is not None: # Check numpy is available
            try:
                # Convert PIL mask to NumPy array
                gradient_mask_np = np.array(gradient_mask_pil)

                # Create a black background clip
                black_bg = ColorClip(size=(width, height), color=(0, 0, 0))
                # Set the gradient (as numpy array) as the mask
                gradient_clip = black_bg.with_mask(ImageClip(gradient_mask_np, is_mask=True))
                gradient_clip = (gradient_clip
                                 .with_duration(article_section_duration)
                                 .with_start(article_start_time)
                                 .with_fps(VIDEO_FPS))
                overlays.append(gradient_clip)
                clip_instances.append(gradient_clip)
            except Exception as e:
                logger.error(f"Failed to create gradient overlay clip: {e}", exc_info=True)
        elif gradient_mask_pil is None:
            logger.warning("Could not create gradient mask PIL image.")
        else: # np is None
             logger.warning("NumPy not available, cannot create gradient overlay mask.")


    # Create Reporter Overlay (circular, bottom-right, during article section)
    if reporter_img_pil and article_section_duration > 0.01:
        logger.info("Creating reporter overlay...")
        reporter_diameter = int(height * REPORTER_SIZE_RATIO)
        reporter_size = (reporter_diameter, reporter_diameter)
        margin_x = int(width * REPORTER_MARGIN_RATIO)
        margin_y = int(height * REPORTER_MARGIN_RATIO)

        try:
            # Resize reporter image and apply circular mask using Pillow
            reporter_img_resized = reporter_img_pil.resize(reporter_size, Image.Resampling.LANCZOS)
            mask = _create_circular_mask(reporter_size)
            if mask and np: # Check numpy is available
                reporter_img_resized.putalpha(mask)
                reporter_np = np.array(reporter_img_resized) # Convert to numpy for ImageClip

                # Position at bottom-right corner
                pos_x = width - reporter_diameter - margin_x
                pos_y = height - reporter_diameter - margin_y
                reporter_clip = (ImageClip(reporter_np, is_mask=False, transparent=True) # Indicate transparency
                                 .with_duration(article_section_duration)
                                 .with_start(article_start_time)
                                 .with_position((pos_x, pos_y)))
                reporter_clip = reporter_clip.with_fps(VIDEO_FPS)
                overlays.append(reporter_clip)
                clip_instances.append(reporter_clip)
            elif not mask:
                logger.warning("Could not create circular mask for reporter.")
            else: # np is None
                logger.warning("NumPy not available, cannot create reporter overlay.")
        except Exception as e:
            logger.error(f"Failed to create reporter overlay: {e}", exc_info=True)


    # Create Title Overlay (bottom-left, during article section)
    if title and article_section_duration > 0.01:
        logger.info("Creating title overlay...")
        title_fontsize = int(height * TITLE_FONT_SIZE_RATIO)
        title_margin_x = int(width * TITLE_MARGIN_RATIO)
        title_margin_y = int(height * TITLE_MARGIN_RATIO)
        # Calculate available width for title, considering reporter overlay
        reporter_width_area = 0
        if reporter_clip:
             # Estimate reporter area width including margins
             reporter_width_area = reporter_diameter + margin_x * 2

        title_max_width = width - title_margin_x * 2 - reporter_width_area
        title_max_width = max(50, title_max_width) # Ensure some minimum width

        try:
            # Use the specified font path
            font_arg = str(font_path) if font_path.exists() else "Arial" # Fallback font
            if font_arg == "Arial": logger.warning(f"Using fallback font 'Arial' for title.")

            # Position at bottom-left corner, adjust vertical position relative to bottom margin
            title_pos_y = height - title_margin_y - title_fontsize * 1.5 # Adjust multiplier as needed

            title_clip = (TextClip(font=font_arg,
                                   text=title, # Corrected argument: txt
                                   font_size=title_fontsize, # Use fontsize instead of font_size
                                   size=(title_max_width, None), # Set max width, height auto
                                   color='white',
                                   stroke_color='black',
                                   stroke_width=1, # Adjusted stroke
                                   method='caption', # Wraps text
                                   text_align='left') # Use align='West' for left alignment
                          .with_duration(article_section_duration)
                          .with_start(article_start_time)
                          .with_position((title_margin_x, title_pos_y)))
            title_clip = title_clip.with_fps(VIDEO_FPS)
            overlays.append(title_clip)
            clip_instances.append(title_clip)
        except Exception as e:
            # Log detailed error, including font path attempt
            logger.error(f"Could not create TextClip for title using font '{font_path}': {e}", exc_info=True)
            logger.error("Ensure ImageMagick is installed and configured correctly, and the font file is valid.")


    # --- 7. Combine Overlays and Set Audio ---
    logger.info("Compositing final video with overlays...")
    final_clip = None # Initialize
    try:
        final_clip = CompositeVideoClip(overlays, size=size)

        # Set the full audio to the final composite clip
        if audio_clip:
            final_clip = final_clip.with_audio(audio_clip)
        # Ensure the final duration matches the audio precisely
        final_clip = final_clip.with_duration(total_duration)
        clip_instances.append(final_clip)
    except Exception as e:
        logger.error(f"Error during final composition or audio setting: {e}", exc_info=True)
        # Close clips before returning
        for clip in clip_instances: clip.close()
        return None


    # --- 8. Write Video File ---
    logger.info(f"Writing video file to: {output_path}")
    final_video_path = None
    try:
        final_clip.write_videofile(
            str(output_path),
            codec='libx264',
            audio_codec='aac',
            fps=VIDEO_FPS,
            preset='medium', # encoding speed/quality trade-off
            threads=os.cpu_count() or 4, # Use available cores or default to 4
            logger='bar' # Show progress bar
        )
        final_video_path = output_path
        logger.info(f"Video created successfully: {output_path}")
    except Exception as e:
        logger.error(f"Error writing video file: {e}", exc_info=True)
        final_video_path = None # Ensure None is returned on failure
    finally:
        # --- 9. Close all clips ---
        logger.debug("Closing MoviePy clips...")
        # Close all tracked clip instances robustly
        for clip in clip_instances:
            if clip:
                try: clip.close()
                except Exception as e: logger.warning(f"Error closing clip {type(clip)}: {e}")

    return final_video_path

# --- Main guard for potential standalone testing ---
if __name__ == "__main__":
    # This section is for basic testing of the utility function
    # It requires manual setup of paths and a dummy Reporter object.
    print("Running video_generator.py standalone for testing.")
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    if not MOVIEPY_AVAILABLE:
        print("Error: MoviePy or dependencies not found. Cannot run test.")
    else:
        # --- Dummy Data for Testing ---
        test_output_dir = Path("./test_video_output")
        test_output_dir.mkdir(exist_ok=True)
        test_filename_base = "test_video_gen"

        # Create dummy assets if they don't exist (replace with actual paths if needed)
        dummy_audio_path = test_output_dir / "dummy_audio.mp3"
        dummy_feat_img_path = test_output_dir / "dummy_feat.png"
        dummy_art_img_path1 = test_output_dir / "dummy_art1.png"
        dummy_art_img_path2 = test_output_dir / "dummy_art2.png"
        dummy_reporter_id = "test_reporter"
        dummy_reporter_img_path = REPORTER_IMAGES_DIR / f"{dummy_reporter_id}.png"

        # Create dummy audio (silence)
        try:
            from pydub import AudioSegment
            silence = AudioSegment.silent(duration=15000) # 15 seconds
            silence.export(dummy_audio_path, format="mp3")
            print(f"Created dummy audio: {dummy_audio_path}")
        except ImportError:
            print("Pydub not installed. Cannot create dummy audio.")
            # Manually place a test audio file at dummy_audio_path if needed
        except Exception as e:
            print(f"Error creating dummy audio: {e}")


        # Create dummy images (simple colored squares)
        def create_dummy_image(path, color, size=(1280, 720)):
            if not Image: return
            try:
                img = Image.new('RGB', size, color=color)
                img.save(path)
                print(f"Created dummy image: {path}")
            except Exception as e:
                print(f"Error creating dummy image {path}: {e}")

        if not dummy_feat_img_path.exists(): create_dummy_image(dummy_feat_img_path, 'blue')
        if not dummy_art_img_path1.exists(): create_dummy_image(dummy_art_img_path1, 'green')
        if not dummy_art_img_path2.exists(): create_dummy_image(dummy_art_img_path2, 'purple')
        # Ensure reporter dir exists
        REPORTER_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
        if not dummy_reporter_img_path.exists(): create_dummy_image(dummy_reporter_img_path, 'red', size=(200, 200))


        # Dummy Reporter
        class DummyTestReporter(Reporter):
             def __init__(self, identifier: str):
                 # super().__init__(identifier) # Call parent if it requires args
                 self.id = identifier # Ensure id is set
                 self.name = identifier.replace("_", " ").title()
                 # self.stable_diffusion_prompt = "" # Add property

        test_reporter = DummyTestReporter(dummy_reporter_id)
        test_title = "This is a Test Video Title for the Overlay - It Should Wrap Correctly and Avoid the Reporter"

        # --- Run Generator ---
        print("\nCalling generate_youtube_video...")
        generated_path = generate_youtube_video(
            reporter=test_reporter,
            title=test_title,
            audio_path=dummy_audio_path,
            featured_image_path=dummy_feat_img_path,
            article_image_paths=[dummy_art_img_path1, dummy_art_img_path2], # Pass as a list
            output_dir=test_output_dir,
            filename_base=test_filename_base,
            # Use default resolution, animation, font, logo
        )

        if generated_path:
            print(f"\n--- Test SUCCESS ---")
            print(f"Video generated at: {generated_path}")
        else:
            print(f"\n--- Test FAILED ---")
            print("Video generation did not complete successfully. Check logs above.")

