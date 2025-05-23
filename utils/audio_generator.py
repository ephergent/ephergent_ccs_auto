#!/usr/bin/env python3
import os
import logging
import re
from pathlib import Path
from dotenv import load_dotenv

# Import OpenAI client library (used for Kokoro API interaction)
try:
    from openai import OpenAI
except ImportError:
    print("Error: openai package not found. Please install it: pip install openai")
    OpenAI = None

# Import pydub for audio effects
try:
    from pydub import AudioSegment
except ImportError:
    print("Error: pydub package not found. Please install it: pip install pydub")
    AudioSegment = None

# Local imports
from utils.reporter import Reporter # Import Reporter for type hinting

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# --- Configuration ---
KOKORO_URL = os.getenv('KOKORO_URL', 'http://localhost:8880/v1') # Kokoro URL from env
ENABLE_AUDIO_EFFECTS = os.getenv('ENABLE_AUDIO_EFFECTS', 'true').lower() in ('true', '1', 'yes', 't')

# Default output directory structure within the main output folder (used by app.py)
AUDIO_SUBDIR = "audio"

# Sound effect paths using pathlib
SOUND_EFFECTS_DIR = Path(__file__).parent.parent / 'sound_effects'
FADE_IN_PATH = SOUND_EFFECTS_DIR / 'fade_in.mp3'
FADE_OUT_PATH = SOUND_EFFECTS_DIR / 'fade_out.mp3'


# --- Kokoro TTS Client Helper Functions (from reference/old_system/tts_api.py) ---

def check_kokoro_available() -> bool:
    """Check if Kokoro service is available"""
    if not OpenAI:
        logger.error("Cannot check Kokoro status: openai package not installed.")
        return False
    if not KOKORO_URL:
        logger.error("KOKORO_URL environment variable not set. Cannot check Kokoro status.")
        return False
    try:
        # Initialize OpenAI client with Kokoro URL
        client = OpenAI(base_url=KOKORO_URL, api_key="not-needed")
        # Try to make a simple request
        client.models.list() # Throws exception if unavailable
        logger.info(f"Kokoro service appears available at {KOKORO_URL}")
        return True
    except Exception as e:
        logger.error(f"Kokoro service not available at {KOKORO_URL}. Is Docker running? Error: {e}")
        return False

def generate_speech(text: str, output_path: Path, voice: str = 'bf_v0isabella(1.5)+bm_v0george(1)+af_v0bella(0.5)+af_v0sarah(0.5)', speed: float = 1.0) -> Path | None:
    """Generate speech from text using Kokoro TTS"""
    if not OpenAI:
        logger.error("Cannot generate speech: openai package not installed.")
        return None
    if not check_kokoro_available():
        logger.error("Kokoro service not available, cannot generate speech.")
        return None
    if not text:
        logger.error("No text provided for speech generation.")
        return None

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Requesting speech generation from Kokoro TTS (Voice: {voice}, Speed: {speed})")
    logger.debug(f"Text length: {len(text)} chars. Output path: {output_path}")

    try:
        # Initialize OpenAI client with Kokoro URL
        client = OpenAI(base_url=KOKORO_URL, api_key="not-needed")

        # Use streaming response to save directly to file
        with client.audio.speech.with_streaming_response.create(
            model="kokoro", # Kokoro model identifier
            voice=voice,
            speed=speed,
            input=text,
            response_format="mp3" # Explicitly set format
        ) as response:
            response.stream_to_file(output_path)

        # Verify file was created and has size
        if output_path.exists() and output_path.stat().st_size > 0:
            logger.info(f"Successfully generated audio via Kokoro: {output_path}")
            return output_path
        else:
            logger.error(f"Kokoro audio file generation failed or resulted in empty file: {output_path}")
            return None

    except Exception as e:
        logger.error(f"Error generating speech with Kokoro API: {e}", exc_info=True)
        return None


# --- Text Preparation ---

def prepare_denizen_text_for_tts(denizen_data: dict, char_backstory: str) -> str:
    """
    Prepares the text from a Dimensional Denizen profile for TTS.
    (Changed first argument to denizen_data dict)

    Args:
        denizen_data (dict): The dictionary containing generated denizen details, including 'name'.
        char_backstory (str): The generated backstory text.

    Returns:
        str: The cleaned text suitable for TTS.
    """
    # Start with the character's name and a brief intro
    name = denizen_data.get('name', 'Unnamed Denizen') # Get name from details if available
    text = f"Today's Dimensional Denizen: {name}. \n\n"

    # Add the backstory
    text += f"Backstory: {char_backstory}\n\n"

    # Optionally add a few key details from the 'details' sub-dict
    char_details = denizen_data.get('details', {})
    text += "Observations:\n"
    if 'appearance' in char_details:
        text += f"- Appearance: {char_details['appearance']}\n"
    if 'char_type' in char_details:
        text += f"- Type: {char_details['char_type']}\n"
    if 'theme' in char_details:
        text += f"- Associated Theme: {char_details['theme']}\n"

    # Clean up potential markdown/HTML remnants (though less likely here)
    text = strip_html_tags(text)
    text = re.sub(r'^\s*#+\s+.*$', '', text, flags=re.MULTILINE)
    text = text.replace('*', '')
    text = re.sub(r'\s*\n\s*', '\n', text)
    text = re.sub(r'[ \t]+', ' ', text)

    # Add a concluding phrase
    text += "\n\n This concludes the profile from The Ephergent."

    logger.info(f"Prepared Denizen text for TTS (Name: {name})")
    return text.strip()


def prepare_text_for_tts(reporter: Reporter, article_content: str, title: str) -> str:
    """
    Cleans and prepares article text for Text-to-Speech conversion.
    (Combines logic from original utils/audio_generator and reference tts_api)

    Args:
        reporter (Reporter): The reporter object containing voice preference.
        article_content (str): The raw article content (potentially with HTML/Markdown).
        title (str): The article title (used for intro).

    Returns:
        str: The cleaned text suitable for TTS.
    """
    if not article_content:
        logger.warning("Article content is empty, TTS output will be minimal.")
        return f"{title}. This has been a broadcast from The Ephergent."
    # Start with the reporter's name and a brief intro
    reporter_name = reporter.name if reporter else "Unknown Reporter"
    text = f"Hello from where ever and when ever you are. I'm {reporter_name} and this is The Ephergent. \n\n"

    # Add the title as an introduction
    text += f"In this neural dimensional broadcast: {title}. \n\n"

    # Strip HTML tags first
    text += strip_html_tags(article_content) # Assumes strip_html_tags is defined below

    # Remove Markdown headings (#, ##, etc.)
    text = re.sub(r'^\s*#+\s+.*$', '', text, flags=re.MULTILINE)

    # Remove specific unwanted phrases (like image captions)
    # ⁂ Video created by The Ephergent's dimensionally-aware AI ⁂
    text = re.sub(r'⁂ Video created by The Ephergent\'s dimensionally-aware AI ⁂', '', text, flags=re.IGNORECASE)
    # ⁂ Audio created by The Ephergent's dimensionally-aware AI [Note: voices may sound different in your dimension.] ⁂
    text = re.sub(r'⁂ Audio created by The Ephergent\'s dimensionally-aware AI \[Note: voices may sound different in your dimension.\] ⁂', '', text, flags=re.IGNORECASE)
    # Your browser does not support the audio element.
    text = re.sub(r'Your browser does not support the audio element.', '', text, flags=re.IGNORECASE)
    # Illustration created by The Ephergent's dimensionally-aware AI ⁂
    text = re.sub(r'Illustration created by The Ephergent\'s dimensionally-aware AI ⁂', '', text, flags=re.IGNORECASE)
    # Remove any other specific phrases (like "Moment Captured by Luminara")
    text = re.sub(r'⁂ Moment Captured by Luminara ⁂', '', text, flags=re.IGNORECASE) # Remove new caption
    # Remove any other specific phrases (like "Moment Captured by Luminara")
    text = re.sub(r'<figure>.*?</figure>', '', text, flags=re.DOTALL | re.IGNORECASE) # Remove figure blocks

    # Remove asterisks used for emphasis
    text = text.replace('*', "'") # Replace asterisks with single quotes
    # Remove horizontal rules
    text = re.sub(r'^\s*---\s*$', '', text, flags=re.MULTILINE)

    # Replace multiple newlines/spaces with single ones for better flow
    text = re.sub(r'\s*\n\s*', '\n', text) # Consolidate newlines
    text = re.sub(r'[ \t]+', ' ', text)    # Consolidate spaces/tabs

    # Add a concluding phrase
    text += "\n\n Thank you for listening to this neural dimensional broadcast of The Ephergent. "

    return text.strip()


# --- Helper Functions (Keep strip_html_tags) ---

def strip_html_tags(text: str) -> str:
    """Removes HTML tags from text using regex."""
    # Simple regex to remove HTML tags
    return re.sub(r'<[^>]*>', '', text)


# --- Audio Effects ---

def combine_audio_with_effects(main_audio_path: Path) -> Path | None:
    """
    Combine the generated audio with fade-in and fade-out effects
    Combines the main generated audio with fade-in and fade-out effects.

    Args:
        main_audio_path (Path): Path to the main TTS generated audio file.

    Returns:
        Path | None: Path to the new combined audio file, or the original path if effects
                     are disabled or fail.
    """
    if not AudioSegment:
        logger.warning("pydub not installed. Skipping audio effects.")
        return main_audio_path
    if not ENABLE_AUDIO_EFFECTS:
        logger.info("Audio effects are disabled via environment variable. Skipping.")
        return main_audio_path

    try:
        # Check if effect files exist (using Path objects)
        if not FADE_IN_PATH.exists():
            logger.warning(f"Fade-in file not found: {FADE_IN_PATH}. Skipping effects.")
            return main_audio_path
        if not FADE_OUT_PATH.exists():
            logger.warning(f"Fade-out file not found: {FADE_OUT_PATH}. Skipping effects.")
            return main_audio_path

        logger.info("Combining audio with fade-in/fade-out effects...")
        # Load audio segments
        fade_in = AudioSegment.from_mp3(FADE_IN_PATH)
        main_audio = AudioSegment.from_mp3(main_audio_path)
        fade_out = AudioSegment.from_mp3(FADE_OUT_PATH)

        # Concatenate
        combined_audio = fade_in + main_audio + fade_out

        # Create new filename for the combined audio
        combined_filename = main_audio_path.stem + "_combined" + main_audio_path.suffix
        combined_path = main_audio_path.with_name(combined_filename)

        # Export the combined audio
        combined_audio.export(combined_path, format="mp3")
        logger.info(f"Combined audio saved to: {combined_path}")

        # Optional: Consider removing the original non-combined file after successful combination
        # try:
        #     main_audio_path.unlink()
        #     logger.info(f"Removed original audio file: {main_audio_path}")
        # except OSError as e:
        #     logger.warning(f"Could not remove original audio file {main_audio_path}: {e}")

        return combined_path

    except Exception as e:
        logger.error(f"Error combining audio with effects: {e}", exc_info=True)
        return main_audio_path # Return original path if combination fails


# --- Main Orchestration Function ---

def generate_article_audio(
    reporter: Reporter,
    article_content: str, # This is the raw content (Markdown/HTML) for standard articles
    title: str,
    output_dir: Path, # Base output directory for the run (e.g., output/run_timestamp)
    filename_base: str,
    speed: float = 1.0
    ) -> Path | None:
    """
    Orchestrates the generation of podcast-style audio from article content using Kokoro TTS.
    Handles text preparation for standard articles.

    Args:
        reporter (Reporter): The reporter object containing voice preference.
        article_content (str): The raw article content (Markdown/HTML).
        title (str): The article title.
        output_dir (Path): The base directory for saving the audio file for this run.
        filename_base (str): A base name for the output file (e.g., 'sanitized_title').
        speed (float): Desired speech speed.

    Returns:
        Path | None: The path to the final generated audio file (potentially combined), or None.
    """
    logger.info(f"--- Starting Kokoro audio generation for: {title[:50]}... ---")

    # 1. Prepare Text (Use standard preparation function)
    audio_text = prepare_text_for_tts(reporter, article_content, title)
    if not audio_text:
        logger.error("Failed to prepare text for TTS.")
        return None

    # 2. Determine Voice (Use reporter's voice directly for Kokoro)
    tts_voice = reporter.voice
    # Add fallback if voice is missing?
    if not tts_voice:
        logger.warning(f"Reporter {reporter.name} has no voice defined. Using default Kokoro voice.")
        # Define a default Kokoro voice if needed
        tts_voice = 'bf_v0isabella(1.5)+bm_v0george(1)+af_v0bella(0.5)+af_v0sarah(0.5)' # Example default
    else:
        logger.info(f"Using reporter voice for Kokoro: {tts_voice}")

    # 3. Define Output Path
    # Place audio in an 'audio' subdirectory within the run's output directory
    audio_output_dir = output_dir / AUDIO_SUBDIR
    audio_filename = f"{filename_base}_audio.mp3"
    raw_audio_path = audio_output_dir / audio_filename

    # 4. Generate Speech using Kokoro
    generated_path = generate_speech(audio_text, raw_audio_path, voice=tts_voice, speed=speed)
    if not generated_path:
        logger.error("Kokoro speech generation failed.")
        return None

    # 5. Combine with Effects (Optional)
    final_audio_path = combine_audio_with_effects(generated_path)

    if final_audio_path:
        logger.info(f"--- Successfully generated Kokoro audio: {final_audio_path} ---")
        return final_audio_path
    else:
        # This case should ideally not happen if generate_speech succeeded and combine failed,
        # as combine_audio_with_effects returns the original path on failure.
        # But log an error just in case.
        logger.error("--- Kokoro audio generation process failed (unexpected state after effects). ---")
        return generated_path # Return the raw generated path if effects somehow failed catastrophically

if __name__ == "__main__":
    print("Testing Audio Generator Utilities...")
    # Ensure Kokoro is running for this test
    if not check_kokoro_available():
        print("Kokoro TTS service is not available. Please start it to run tests.")
    else:
        print("Kokoro service is available.")
        # Dummy data for testing
        test_output_dir = Path("./output/output_temp_audio_test")
        test_output_dir.mkdir(parents=True, exist_ok=True) # Ensure parent dirs for audio subdir
        
        test_article_content = "Hello world. This is a test of the Ephergent audio generation system. Will it blend? Let's find out. This text includes <html>tags</html> and *markdown*."
        test_title = "Audio Test Title"
        test_filename_base = "test_audio_output"
        # test_voice = 'bf_v0isabella(1.5)+bm_v0george(1)+af_v0bella(0.5)+af_v0sarah(0.5)' # Example Kokoro voice string
        # test_speed = 1.1

        # 0. Get a reporter instance (if needed)
        reporter = Reporter(identifier="pixel_paradox") # Assuming 'pixel_paradox' exists

        # 1. Test text preparation
        print("\nTesting text preparation...")
        prepared_text = prepare_text_for_tts(reporter, test_article_content, test_title)
        print(f"Prepared text:\n{prepared_text}")

        # 2. Test speech generation using generate_article_audio
        print("\nTesting full audio generation (generate_article_audio)...")
        # Create a dummy reporter for testing
        try:
            test_reporter = Reporter(identifier="pixel_paradox") # Assuming 'pixel_paradox' exists
            if not test_reporter.reporter_data: # Check if reporter loaded successfully
                 print("Warning: Could not load 'pixel_paradox' reporter. Using default voice for test.")
                 # Fallback to a generic Reporter instance if specific one fails
                 class DummyReporter:
                     name = "Test Reporter"
                     voice = 'bf_v0isabella(1.5)+bm_v0george(1)+af_v0bella(0.5)+af_v0sarah(0.5)' # Default Kokoro voice
                 test_reporter = DummyReporter()
        except Exception as e:
            print(f"Error initializing Reporter: {e}. Using default voice for test.")
            class DummyReporter: # Fallback
                name = "Test Reporter"
                voice = 'bf_v0isabella(1.5)+bm_v0george(1)+af_v0bella(0.5)+af_v0sarah(0.5)'
            test_reporter = DummyReporter()


        final_audio_path = generate_article_audio(
            reporter=test_reporter,
            article_content=test_article_content,
            title=test_title,
            output_dir=test_output_dir, # Pass base output dir
            filename_base=test_filename_base,
            speed=1.0 # Default speed
        )

        if final_audio_path:
            print(f"Full audio generation successful: {final_audio_path}")
        else:
            print("Full audio generation failed.")


        # Test Denizen text prep
        print("\nTesting Denizen text preparation...")
        dummy_denizen_data = {
            'name': 'Zorp Glorbax',
            'details': {
                'appearance': 'Crystalline', 'char_type': 'Thought Harvester',
                'theme': 'Cosmic Horror The Edge'
            }
        }
        dummy_backstory = "Zorp collects forgotten memories from dying stars."
        prepared_denizen_text = prepare_denizen_text_for_tts(dummy_denizen_data, dummy_backstory)
        print(f"Prepared Denizen text:\n{prepared_denizen_text}")


        # Clean up
        import shutil
        if test_output_dir.exists():
            print(f"\nCleaning up test directory: {test_output_dir}")
            shutil.rmtree(test_output_dir)
