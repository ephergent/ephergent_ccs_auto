#!/usr/bin/env python3
import os
import json
import random
import logging
import requests
import uuid
import time
import re
from pathlib import Path
from urllib.parse import urlencode
from dotenv import load_dotenv
import websocket # Use the websocket-client package

# Try to import genai, handle import error if needed
try:
    import google.generativeai as genai
except ImportError:
    print("Error: google-generativeai package not found. Please install it: pip install google-generativeai")
    genai = None # Set to None if import fails

from datetime import datetime # Added for standalone test block timestamp

# Local imports
# Assuming utils is in the same parent directory or PYTHONPATH is set correctly
try:
    from utils.system_prompt import load_system_prompt
    from utils.pelican_exporter import sanitize_filename # For filename sanitization
    # Import Reporter only needed for the standalone test block now
    from utils.reporter import Reporter
    # Import image generation function from image_generator
    from utils.image_generator import generate_image_with_comfyui, DEFAULT_WORKFLOW_FILE
except ImportError as e:
    print(f"Error: Could not import utility functions: {e}. Make sure utils directory is accessible.")
    # Define dummy functions if utils are missing, to allow basic script structure check
    def load_system_prompt(): return "Dummy system prompt."
    def sanitize_filename(text, allow_unicode=False): return re.sub(r'[^\w\-]+', '_', text).strip('_').lower() or "sanitized_name"
    # Dummy image generator function
    def generate_image_with_comfyui(prompt: str, workflow_filepath: Path, output_dir: Path, filename: str) -> Path | None:
        logger = logging.getLogger(__name__) # Need logger inside dummy
        logger.warning("Using dummy image generator function.")
        dummy_path = output_dir / filename
        try:
            dummy_path.touch() # Create empty file
            return dummy_path
        except Exception:
            return None
    DEFAULT_WORKFLOW_FILE = Path("dummy_workflow.json") # Dummy path
    class Reporter: # Dummy Reporter
        def __init__(self, *args, **kwargs): pass


# --- Basic Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('profile_generator.log'), # Log to file
        logging.StreamHandler() # Also log to console
    ]
)
logger = logging.getLogger(__name__)
load_dotenv()

# --- Gemini Client Initialization ---
GEMINI_MODEL = os.getenv('GEMINI_MODEL', 'gemini-2.0-flash') # Updated default model
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
gemini_client = None
if genai and GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_client = genai.GenerativeModel(GEMINI_MODEL)
        logger.info(f"Gemini client initialized successfully with {GEMINI_MODEL}.")
    except Exception as e:
        logger.error(f"Failed to initialize Gemini client: {e}")
elif not genai:
    logger.error("Gemini client cannot be initialized (package missing).")
elif not GEMINI_API_KEY:
    logger.error("Gemini client cannot be initialized (API key missing).")


# --- ComfyUI Configuration ---
# COMFYUI_URL is used implicitly by generate_image_with_comfyui from image_generator.py
# DEFAULT_WORKFLOW_FILE is imported from image_generator.py

# --- Ephergent Universe Character Details (Inspired by System Prompt) ---
dimension_themes = [
    "Urban Sci-Fi Prime Material", "Gothic Horror Nocturne", "Steampunk Cogsworth",
    "Ecological Sci-Fi Verdantia", "Cosmic Horror The Edge",
    "Absurdist Bureaucracy", "Political Thriller",
    "Reality Stabilization", "Narrative Causality", "Interdimensional Economics", "Market Volatility",
    "Data-Driven Divination", "Memetic Warfare"
]
character_types = [
    "human", "dimensional hybrid", "sentient algorithm", "reality architect", "thought harvester",
    "paradox entity", "frequency being", "dream cartographer", "cosmic bureaucrat",
    "metamorphic collective", "cybernetically enhanced dinosaur", "telepathic houseplant",
    "sentient weather pattern", "rogue AI assistant", "memory broker", "conceptual artist",
    "bio-hacker", "quantum shifter", "void nomad", "emotion sculptor",
    "probability weaver", "data specter", "sentient stapler", "AI espresso machine"
]
appearance_traits = [
    "crystalline", "shadow-shifting", "bioluminescent", "flame-core", "void-touched",
    "techno-organic", "multi-phased", "gravity-defying", "time-fractured", "echo-resonant",
    "quantum-patterned", "dream-woven", "dimension-spliced", "frequency-vibrant", "reality-warped",
    "neural-networked", "probability-fluctuating", "cosmic-infused", "glitch-stitched", "sound-sculpted",
    "holographic", "nanite-swarm", "data-corrupted", "bio-luminescent fungus", "clockwork",
    "composed of flowing light ribbons", "obsidian-like", "woven from pure information",
    "metallic orange (stapler)", "glowing blue AI core", "ancient tree bark", "pale gothic"
]
hair_colors = ["red", "blonde", "black", "blue", "green", "purple", "white", "silver", "pink", "iridescent", "data-stream", "static-charged", "color-shifting", "molten metal", "nebula-patterned", "transparent", "obsidian-black", "platinum blonde to rose gold gradient", "fiber optic cables"]
hair_types = ["long", "short", "curly", "straight", "wavy", "spiky", "braided", "dreadlocked", "geometric", "energy-based", "fiber-optic", "no", "crystalline shards", "liquid data", "temporal loops", "impossible geometric loop", "severe undercut", "mohawk", "flowing"]
eye_colors = ["blue", "green", "brown", "hazel", "amber", "gray", "red", "purple", "black", "glowing", "nebula-filled", "multi-faceted", "static-filled", "void-like", "barcode-scan", "clockface", "pixelated", "miniature swirling galaxies", "tiny hourglasses", "cybernetic lenses", "augmented reality display", "optical sensors (stapler)"]
clothing_styles = [
    "cyberpunk gear", "steampunk attire", "lab coat", "dimensional traveler rags",
    "formal suit made of void", "glam rock outfit", "utilitarian jumpsuit",
    "robes woven from time", "armor plated with paradoxes", "sentient fabric",
    "holographic projection", "scrap-metal couture", "bio-integrated fashion",
    "high-tech journalist gear", "high-collared white lab coat with circuit patterns",
    "distressed dark tech-punk jacket", "impeccably tailored shimmering suit",
    "flamboyant jacket of solidified sound waves", "avant-garde outfit blending eras",
    "sleek AI core housing", "practical hazard-proof gear", "classic academic tweed jacket",
    "dynamic cloak woven from information", "dark hooded cloak with scavenged tech",
    "elaborate dark gothic attire"
]
accessories = [
    "cybernetic enhancements", "floating data-slate", "reality-stabilizing amulet", "temporal compass",
    "memetic filter glasses", "probability manipulator gauntlet", "sentient scarf", "portable wormhole generator",
    "anti-paradox charm", "neural interface plugs", "tool belt with impossible tools", "pet miniature black hole",
    "multifunctional collar", "holographic press badge", "geometric jewelry", "oddly shaped coffee cup",
    "embedded scientific tools", "augmented reality glasses", "cybernetic interface hand",
    "amulet with dark jewel", "futuristic microphone drone", "glowing data-jewels",
    "jewelry like frozen moments", "holographic interfaces", "multi-lensed camera rig",
    "round goggles with holographic lenses", "glowing artifact", "visor displaying probability streams"
]
expressions = [
    "wry expression", "quirky smile", "confused look", "mischievous grin",
    "stoic gaze", "playful smirk", "intense focus", "absurdly serious expression",
    "knowing glance", "perpetually surprised", "calmly observing chaos", "slightly glitching smile",
    "contemplating a paradox", "adjusting reality-stabilizing amulet", "sipping cosmic coffee",
    "serious analytical expression", "intense and focused", "calculating and enigmatic",
    "charismatic and mischievous", "graceful and editorial", "detached fascination to horror",
    "intense concentration", "smug determined sneaky", "wise and ancient", "sophisticated weariness"
]
mundane_actions = [
    "reading a data slate", "adjusting their clothing", "looking thoughtfully into the distance",
    "examining a strange artifact", "operating a peculiar device", "drinking a futuristic beverage",
    "waiting for a dimensional bus", "polishing their accessory", "consulting a star chart",
    "typing on a holographic keyboard", "repairing a small gadget", "watering a bizarre plant",
    "sorting through interdimensional mail", "calibrating a sensor", "sketching in a notebook",
    "brewing espresso", "looking through a camera viewfinder", "holding a complex data slate",
    "scuttling across a desk", "issuing telepathic warnings", "observing a rift with dismay"
]

# --- Helper Functions ---

def generate_character_details() -> dict:
    """Generates a dictionary of random character details."""
    hair_detail = f"{random.choice(hair_types)} {random.choice(hair_colors)}" if random.choice([True, False]) else f"{random.choice(hair_colors)}"
    if "no" in hair_detail: hair_detail = "no hair"

    details = {
        "appearance": random.choice(appearance_traits),
        "char_type": random.choice(character_types),
        "gender": random.choice(["woman", "man", "entity", "being", "construct", "collective"]),
        "hair": hair_detail,
        "eyes": random.choice(eye_colors),
        "clothing": random.choice(clothing_styles),
        "accessory": random.choice(accessories),
        "expression": random.choice(expressions),
        "theme": random.choice(dimension_themes)
    }
    logger.info(f"Generated character details: {details}")
    return details

def generate_name_and_backstory(character_details: dict, system_prompt: str) -> tuple[str, str]:
    """Generates a character name and backstory using Gemini."""
    if not gemini_client:
        logger.error("Gemini client not available. Cannot generate name and backstory.")
        return "Unnamed Character", "No backstory generated (Gemini unavailable)."

    # Construct a description string from details
    desc = (
        f"An Ephergent universe character: a {character_details['appearance']} {character_details['char_type']} "
        f"{character_details['gender']} with {character_details['hair']} hair and {character_details['eyes']} eyes. "
        f"They wear {character_details['clothing']} and have a {character_details['accessory']}. "
        f"Their expression is {character_details['expression']}. Associated theme: {character_details['theme']}."
    )

    prompt_instruction = f"""
{system_prompt}

Character Description Seed:
{desc}

TASK:
Based *only* on the character description seed provided above and the Ephergent universe context from the system prompt:
1. Generate a unique, evocative, and fitting name for this character. The name should sound like it belongs in a quirky, absurd sci-fi/fantasy universe. Output *only* the name on the first line.
2. Generate a short, intriguing backstory (2-4 sentences) for this character, hinting at their role or history within the Ephergent universe. Output *only* the backstory starting on the second line.

Example Output Format:
Zorp Glorbax
Zorp runs a pawn shop specializing in slightly used paradoxes, always looking for the next big temporal loophole to exploit. He claims his third eye is just a birthmark, but no one believes him.
"""
    try:
        logger.info("Generating name and backstory via Gemini...")
        # Pass instruction directly, assuming system prompt provides general context
        response = gemini_client.generate_content(prompt_instruction)

        # Check for safety ratings and blocked prompts before accessing text
        if response.prompt_feedback.block_reason:
            logger.error(f"Gemini prompt blocked for name/backstory generation. Reason: {response.prompt_feedback.block_reason}")
            return "Generation Error", f"Failed to generate: Prompt blocked ({response.prompt_feedback.block_reason})"

        # Access text safely
        generated_text = ""
        if response.parts:
            generated_text = response.parts[0].text
        elif hasattr(response, 'text'): # Fallback for older versions or different response structures
            generated_text = response.text
        else:
            logger.error("Gemini response for name/backstory missing text attribute/parts.")
            return "Generation Error", "Failed to get response text from Gemini."

        lines = generated_text.strip().split('\n', 1)
        name = lines[0].strip()
        backstory = lines[1].strip() if len(lines) > 1 else "No backstory generated."

        # Basic validation
        if not name or len(name) > 50: # Arbitrary length check
             logger.warning(f"Generated name seems invalid or too long: '{name}'. Using fallback.")
             name = f"{random.choice(appearance_traits).capitalize()} {random.choice(character_types).capitalize()}" # Simple fallback name
        if not backstory or len(backstory) < 10: # Arbitrary length check
             logger.warning(f"Generated backstory seems invalid or too short: '{backstory}'. Using fallback.")
             backstory = f"{name} exists within the Ephergent universe, navigating its peculiar realities."

        logger.info(f"Generated Name: {name}")
        logger.info(f"Generated Backstory: {backstory}")
        return name, backstory

    except Exception as e:
        logger.error(f"Error generating name/backstory via Gemini: {e}", exc_info=True)
        # Check if the error is related to API key or configuration
        if "API key not valid" in str(e):
             logger.error("Please check your GEMINI_API_KEY environment variable.")
        return "Generation Error", f"Failed to generate backstory due to error: {e}"

def generate_comfyui_image_prompt(character_details: dict, character_name: str, image_type: str) -> str:
    """Constructs the image prompt for ComfyUI based on image type."""
    # Base description is the same
    character_desc = (
        f"a {character_details['appearance']} {character_details['char_type']} {character_details['gender']} "
        f"named {character_name}, with {character_details['hair']} hair and {character_details['eyes']} eyes, "
        f"dressed in {character_details['clothing']}, featuring a {character_details['accessory']}."
    )

    if image_type == "profile":
        prompt_text = f"""
        A digitally illustrated drawing in anime manga comic book style. Professional headshot portrait.
        Subject: {character_desc}. The character has a {character_details['expression']}.
        Setting: Abstract background incorporating elements of the '{character_details['theme']}' theme, possibly with subtle dimensional distortions or energy effects. Include the character's name '{character_name}' into the background elements (e.g., graffiti, holographic text, energy patterns).
        Lighting: Dramatic, high-contrast lighting that highlights the character's features and textures.
        Composition: Centered portrait, medium shot (head and shoulders or waist up).
        Focus: Sharp focus on the character, background slightly blurred.
        Overall Mood: Evocative of the Ephergent universe - quirky, slightly absurd, sci-fi/fantasy blend. High quality, detailed illustration.
        """
        logger.info(f"Generated PROFILE image prompt: {prompt_text[:150]}...")
    elif image_type == "action":
        action = random.choice(mundane_actions)
        prompt_text = f"""
        A digitally illustrated drawing in anime manga comic book style.
        Subject: {character_desc}. The character is captured {action}. Their expression is {character_details['expression']}.
        Setting: A mundane but slightly strange Ephergent universe location related to the '{character_details['theme']}' theme (e.g., a bizarre market stall, a glitching waiting room, a workshop with impossible tools, a street corner with odd architecture).
        Lighting: Naturalistic but interesting lighting suitable for the scene.
        Composition: Full body or three-quarter shot showing the character interacting with their environment or performing the action. Dynamic angle if appropriate.
        Focus: Sharp focus on the character and their immediate action, background showing context but slightly less sharp.
        Overall Mood: A slice-of-life moment in the quirky, absurd Ephergent universe. High quality, detailed illustration.
        """
        logger.info(f"Generated ACTION image prompt: {prompt_text}")
    else:
        logger.error(f"Unknown image type requested: {image_type}")
        return "" # Return empty string for error

    # Clean up whitespace and ensure single lines for ComfyUI
    return " ".join([line.strip() for line in prompt_text.strip().splitlines() if line.strip()])


# --- ComfyUI API Functions (REMOVED) ---
# These functions (open_websocket_connection, queue_prompt, get_image, get_history, track_progress, get_images_from_history)
# are now centralized in utils/image_generator.py.
# We will use the imported generate_image_with_comfyui function instead.


# --- Adapted ComfyUI Generation Function (REPLACED) ---
# Replaced with direct call to imported generate_image_with_comfyui

def generate_denizen_markdown(
    char_name: str,
    char_backstory: str,
    char_details: dict,
    profile_image_filename: str | None, # Just the filename.png
    action_image_filename: str | None, # Just the filename.png
    profile_image_prompt: str,
    action_image_prompt: str
    ) -> str:
    """Generates the full markdown content for the Denizen profile."""

    # Use relative paths suitable for Pelican export (images/filename.ext)
    # These will be converted to {static}/images/filename.ext by the exporter
    profile_img_md_path = f"images/{profile_image_filename}" if profile_image_filename else ""
    action_img_md_path = f"images/{action_image_filename}" if action_image_filename else ""

    # --- Build Details Section ---
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

    details_section = details_md + "\n".join(details_list) if details_list else ""


    # --- Assemble Full Markdown ---
    markdown_content = f"""
## TODAY'S DIMENSIONAL DENIZEN

**Generated Profile Image:** {profile_img_md_path if profile_img_md_path else 'Generation Failed'}

![{char_name} Profile]({profile_img_md_path})

### {char_name.upper()}

{details_section}

---

### BACKSTORY

{char_backstory}

---

### OBSERVATIONS

**Generated Action Image:** {action_img_md_path if action_img_md_path else 'Generation Failed'}

![{char_name} Action Shot]({action_img_md_path})

*A recent observation captured {char_name} engaging in a moment of mundane activity within the Ephergent.*

---

### GENERATION DETAILS

#### Profile Image Prompt:
```
{profile_image_prompt}
```

#### Action Image Prompt:
```
{action_image_prompt}
```
"""
    return markdown_content.strip()


# --- Main Function (Callable) ---
def generate_denizen_profile(output_dir: Path) -> dict | None:
    """
    Generates a Dimensional Denizen profile including name, backstory,
    profile image, action image, and markdown content.

    Args:
        output_dir (Path): The base output directory for the current run
                           (e.g., output/run_YYYYMMDD_HHMMSS). Artifacts
                           will be saved within subdirectories here.

    Returns:
        dict | None: A dictionary containing generated details:
                     'name', 'backstory', 'details' (dict),
                     'profile_image_path' (Path | None),
                     'action_image_path' (Path | None),
                     'markdown_content' (str | None), # Raw markdown body
                     'filename_base' (str)
                     Returns None on critical failure.
    """
    logger.info("--- Starting Dimensional Denizen Profile Generation ---")

    # Ensure required clients/config are available
    if not gemini_client:
        logger.error("Gemini client is not initialized. Cannot generate name/backstory. Aborting Denizen generation.")
        return None
    # COMFYUI_URL check is implicit in generate_image_with_comfyui
    if not DEFAULT_WORKFLOW_FILE.exists():
         logger.error(f"Default ComfyUI workflow file not found at '{DEFAULT_WORKFLOW_FILE}'. Aborting Denizen generation.")
         return None

    # Create specific subdirectories within the run's output dir
    denizen_img_output_dir = output_dir / "images" # Images go here
    denizen_md_output_dir = output_dir # Markdown saved directly in run dir for now
    denizen_img_output_dir.mkdir(parents=True, exist_ok=True)
    denizen_md_output_dir.mkdir(parents=True, exist_ok=True)

    profile_image_path = None
    action_image_path = None
    markdown_content = None

    try:
        # 1. Load System Prompt (using Pixel Paradox context)
        logger.info("Loading system prompt for Denizen generation...")
        # Load the 'pixel_paradox' system prompt directly
        system_prompt = load_system_prompt()
        if not system_prompt:
            logger.error("Failed to load system prompt. Aborting Denizen generation.")
            # Provide a basic fallback if loading fails but log error
            # system_prompt = "You are generating content for the Ephergent universe, a quirky sci-fi/fantasy setting focused on technology, retro aesthetics, and absurdity. Focus on generating a unique character name and short backstory."
            return None # Abort if prompt is critical

        logger.info("Using Ephergent system prompt context.")

        # 2. Generate Character Details
        logger.info("Generating character details...")
        char_details = generate_character_details()

        # 3. Generate Name and Backstory
        logger.info("Generating name and backstory...")
        char_name, char_backstory = generate_name_and_backstory(char_details, system_prompt)
        if char_name == "Generation Error":
            logger.error("Failed to generate name and backstory. Aborting Denizen generation.")
            return None

        # Add name to details dict for convenience
        char_details['name'] = char_name

        # Sanitize name for filenames (use a 'denizen_' prefix for clarity)
        safe_name = sanitize_filename(char_name)
        filename_base = f"denizen_{safe_name}" # e.g., denizen_zorp_glorbax

        # 4. Generate Image Prompts
        logger.info("Generating image prompts...")
        profile_image_prompt = generate_comfyui_image_prompt(char_details, char_name, "profile")
        action_image_prompt = generate_comfyui_image_prompt(char_details, char_name, "action")

        # 5. Generate Profile Image using imported function
        logger.info("Generating profile image via ComfyUI...")
        profile_filename = f"{filename_base}_profile.png" # Define filename
        profile_image_path = generate_image_with_comfyui(
            prompt=profile_image_prompt,
            workflow_filepath=DEFAULT_WORKFLOW_FILE,
            output_dir=denizen_img_output_dir, # Save to run's images subdir
            filename=profile_filename
        )
        if not profile_image_path:
            logger.warning("Failed to generate profile image.") # Non-fatal, continue

        # 6. Generate Action Image using imported function
        logger.info("Generating action image via ComfyUI...")
        action_filename = f"{filename_base}_action.png" # Define filename
        action_image_path = generate_image_with_comfyui(
            prompt=action_image_prompt,
            workflow_filepath=DEFAULT_WORKFLOW_FILE,
            output_dir=denizen_img_output_dir, # Save to run's images subdir
            filename=action_filename
        )
        if not action_image_path:
            logger.warning("Failed to generate action image.") # Non-fatal, continue

        # 7. Generate Markdown Content (String) - This is the BODY content
        logger.info("Generating markdown content...")
        markdown_content = generate_denizen_markdown(
            char_name=char_name,
            char_backstory=char_backstory,
            char_details=char_details,
            profile_image_filename=profile_image_path.name if profile_image_path else None,
            action_image_filename=action_image_path.name if action_image_path else None,
            profile_image_prompt=profile_image_prompt,
            action_image_prompt=action_image_prompt
        )

        # Optional: Save markdown to a temporary file in the run dir for debugging/archiving
        # This saved file contains only the body, not the full Pelican markdown
        md_temp_filename = f"{filename_base}_body.md"
        md_temp_path = denizen_md_output_dir / md_temp_filename
        try:
            with open(md_temp_path, "w", encoding="utf-8") as f:
                f.write(markdown_content)
            logger.info(f"Saved temporary Denizen markdown body to: {md_temp_path}")
        except IOError as e:
            logger.error(f"Failed to save temporary Denizen markdown file: {e}")
            # Don't fail the whole process, just log error

        logger.info("--- Dimensional Denizen Profile Generation Finished Successfully ---")

        return {
            "name": char_name,
            "backstory": char_backstory, # Use backstory as summary for newsletter/Pelican
            "details": char_details, # Pass the full details dict
            "profile_image_path": profile_image_path, # Path object or None
            "action_image_path": action_image_path, # Path object or None
            "markdown_content": markdown_content, # Raw markdown body string or None
            "filename_base": filename_base # e.g., denizen_zorp_glorbax
        }

    except Exception as e:
        logger.error(f"An unexpected error occurred during Denizen profile generation: {e}", exc_info=True)
        return None # Indicate failure


# --- Main Execution (for standalone testing) ---
if __name__ == "__main__":
    logger.info("--- Running Ephergent Profile Generator Standalone Test ---")
    # Create a dummy output directory for the test run
    test_output_dir = Path(__file__).parent.parent / "output" / f"test_run_denizen_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    test_output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Using test output directory: {test_output_dir}")

    # Check prerequisites
    if not GEMINI_API_KEY:
        logger.error("GEMINI_API_KEY not found in environment variables. Cannot generate name/backstory. Aborting test.")
        exit()
    if not os.getenv('COMFYUI_URL'):
        logger.error("COMFYUI_URL not found in environment variables. Cannot connect to ComfyUI. Aborting test.")
        exit()
    if not DEFAULT_WORKFLOW_FILE.exists():
        logger.error(f"Default workflow file not found at {DEFAULT_WORKFLOW_FILE}. Aborting test.")
        exit()

    denizen_data = generate_denizen_profile(test_output_dir)

    if denizen_data:
        logger.info("--- Standalone Test Results ---")
        logger.info(f"Name: {denizen_data['name']}")
        logger.info(f"Backstory: {denizen_data['backstory'][:100]}...")
        logger.info(f"Profile Image Path: {denizen_data['profile_image_path']}")
        logger.info(f"Action Image Path: {denizen_data['action_image_path']}")
        logger.info(f"Filename Base: {denizen_data['filename_base']}")
        logger.info(f"Markdown Content Generated: {'Yes' if denizen_data['markdown_content'] else 'No'}")
        # logger.info(f"Markdown Preview:\n{denizen_data['markdown_content'][:500]}\n...")
    else:
        logger.error("Standalone Denizen generation test failed.")

    # Optional: Add cleanup for the test directory if desired
    # import shutil
    # logger.info(f"Cleaning up test directory: {test_output_dir}")
    # shutil.rmtree(test_output_dir)
