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
from typing import Dict, Any, Optional, List, Tuple # Import typing helpers

# Try to import genai, handle import error if needed
try:
    import google.generativeai as genai
    from google.generativeai.types import GenerationConfig # For JSON mode
except ImportError:
    print("Error: google-generativeai package not found. Please install it: pip install google-generativeai")
    genai = None # Set to None if import fails
    GenerationConfig = None # Set to None if import fails


# Local imports
from utils.reporter import Reporter
from utils.system_prompt import load_system_prompt # Import the refactored function
# Imports needed for the updated __main__ block
# from utils.topic_generator import generate_topic # No longer used in main flow
# from utils.title import generate_titles # No longer used in main flow
# from utils.article import generate_article # No longer used in main flow
from utils.pelican_exporter import sanitize_filename

# Set up logging
LOG_FILE_PATH = Path(__file__).parent.parent / 'ephergent_content_creator.log'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE_PATH), # Log to file
        logging.StreamHandler() # Also log to console
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# --- Gemini Client Initialization ---
GEMINI_MODEL = os.getenv('GEMINI_MODEL', 'gemini-2.0-flash') # Use a model supporting JSON output if possible
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
gemini_client = None
if genai and GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        # Configure for JSON output if the model supports it
        generation_config = None
        if GenerationConfig:
             try:
                 # Attempt to configure for JSON output
                 generation_config = GenerationConfig(response_mime_type="application/json")
                 logger.info("Configuring Gemini client for JSON output.")
             except ValueError: # Model might not support JSON output
                 logger.warning(f"Model {GEMINI_MODEL} might not support JSON output directly. Will parse text response.")
                 generation_config = None # Fallback to default text output

        gemini_client = genai.GenerativeModel(
            GEMINI_MODEL,
            generation_config=generation_config # Pass config here (will be None if JSON not supported)
        )
        logger.info(f"Gemini client initialized successfully with {GEMINI_MODEL}.")
    except Exception as e:
        logger.error(f"Failed to initialize Gemini client for summarization: {e}")
elif not genai:
    logger.error("Gemini client cannot be initialized for summarization (package missing).")
elif not GEMINI_API_KEY:
    logger.error("Gemini client cannot be initialized for summarization (API key missing).")

# --- ComfyUI Configuration ---
COMFYUI_URL = os.getenv('COMFYUI_URL', 'http://127.0.0.1:8188') # Default to local ComfyUI
DEFAULT_WORKFLOW_FILE = Path(__file__).parent.parent / 'text_to_image_1lora_FLUX.json' # Assumes workflow in project root

# --- Image Prompt Generation ---

# Define the base style prefix globally
STYLE_PREFIX = "A digitally illustrated drawing in anime manga comic book style,"
PROMPTS_JSON_PATH = Path(__file__).parent.parent / 'prompts' / 'personality_prompts.json'

def generate_article_essence_image_prompt(
    story_data: Dict[str, Any],
    reporter: Reporter
) -> str | None:
    """
    Generates a single ComfyUI prompt for an article essence image using an LLM.
    The prompt aims to create one image that captures the overall mood or a key concept
    of the entire article, like a striking feature photo for a magazine article.

    Args:
        story_data (Dict[str, Any]): Dictionary containing story details (title, content, etc.).
        reporter (Reporter): The reporter object associated with the story.

    Returns:
        str | None: The generated image prompt string, or None on failure.
    """
    if not gemini_client:
        logger.error("Gemini client needed for article essence prompt generation but is not available.")
        return None

    title = story_data.get('title', 'Untitled Report')
    article_content = story_data.get('content', '')
    location = story_data.get('location', 'Unknown Location')

    if not article_content:
        logger.warning(f"Article content is empty for title '{title}'. Prompt generation might lack context.")
        # Allow generation to proceed, but quality might be lower

    # --- Load known character names and detect mentions ---
    # Keep this logic as mentioned characters might be relevant for the essence image
    known_character_names = []
    try:
        if PROMPTS_JSON_PATH.exists():
            with open(PROMPTS_JSON_PATH, 'r', encoding='utf-8') as f:
                reporters_data = json.load(f)
            known_character_names = [
                r.get('name') for r in reporters_data.get('reporters', []) if r.get('name')
            ]
            logger.info(f"Loaded {len(known_character_names)} known character names for mention detection.")
        else:
            logger.warning(f"Personality prompts file not found at {PROMPTS_JSON_PATH}. Cannot detect character mentions.")
    except (json.JSONDecodeError, IOError) as e:
        logger.error(f"Error loading or parsing {PROMPTS_JSON_PATH}: {e}. Cannot detect character mentions.")

    # Detect mentioned characters in the article content (case-insensitive)
    mentioned_characters = []
    if known_character_names and article_content:
        article_content_lower = article_content.lower()
        for name in known_character_names:
            # Use regex for better word boundary matching
            # Pattern: \b(name)\b - matches whole word, case-insensitive
            pattern = r'\b(' + re.escape(name.lower()) + r')\b'
            if re.search(pattern, article_content_lower):
                 mentioned_characters.append(name)

    character_mention_info = ""
    if mentioned_characters:
        # Remove duplicates just in case
        mentioned_characters = sorted(list(set(mentioned_characters)))
        character_mention_info = (
            f"Characters Mentioned: The following known Ephergent characters were mentioned in the article: "
            f"{', '.join(mentioned_characters)}. If the essence of the article involves one of these characters, "
            f"depict them accurately based on their known appearance (if available, otherwise interpret creatively within the Ephergent style)."
        )
        logger.info(f"Detected mentions of characters: {', '.join(mentioned_characters)}")
    else:
        character_mention_info = "Characters Mentioned: No specific known Ephergent characters were detected in the article text."
        logger.info("No known character mentions detected in the article.")
    # --- End of character detection block ---

    # Extract universe themes/elements (keep this)
    dimension_themes = [
        "Urban Sci-Fi Prime Material", "Gothic Horror Nocturne", "Steampunk Cogsworth",
        "Cyberpunk AI Mechanica", "Ecological Sci-Fi Verdantia", "Cosmic Horror The Edge",
        "Time-Travel Mystery Chronos Reach", "Absurdist Bureaucracy", "Political Thriller",
        "Reality Stabilization", "Narrative Causality"
    ]
    universe_elements = [
        "The Ephergent HQ", "A1 AI Assistant", "CLX Crystallized Laughter", "Those Who Wait",
        "Reality Anchors", "Narrative Engine", "Dimensional Rifts", "Temporal Paradoxes",
        "Sentient Infrastructure", "Reality Glitches", "Void Incursions", "Anti-Creation",
        "Great Thought-Root Network", "Cybernetic Dinosaurs", "Gravity Reversals",
        "Multi-directional Time Flow", "Probability Storms", "Data Streams", "Holographic Interfaces",
        "Quantum Computing", "Espresso Machine Robot", "Dimensional Barriers", "Forbidden Knowledge",
        "Cyclical Collapse", "Reality Fatigue", "Rogue Narrative Loops"
    ]
    visual_keywords = [
        "unpredictable physics", "obsidian architecture", "stained-glass observatories", "clockwork mechanisms",
        "artisanal tea", "neon-lit megacities", "telepathic plants", "glowing root networks",
        "half-formed reality", "time loops", "fragmented memories", "shifting geometric patterns",
        "flowing streams of light", "ancient symbols", "star charts", "data constructs", "cosmic void",
        "glitching displays", "unstable energy fluctuations", "corrupted data", "fractal patterns",
        "impossible geometries", "patchwork realities", "shadowy figures", "metallic tang of ozone",
        "cascading waterfalls (up and down)", "high-contrast lighting", "flickering neon", "shadowy ambiance"
    ]

    selected_theme = random.choice(reporter.topics) if reporter.topics else random.choice(dimension_themes)
    if not isinstance(selected_theme, str):
        selected_theme = random.choice(dimension_themes)

    num_elements = random.randint(1, 3)
    selected_elements = random.sample(universe_elements + visual_keywords, k=num_elements)

    prompt_instruction = f"""
    Persona Context: You are assisting {reporter.name}, the {reporter.topics[1] if reporter.topics else 'interdimensional'} correspondent for 'The Ephergent'.
    Reporter Persona Snippet: {reporter.prompt[:500]}...
    Article Title: {title}
    Article Location: {location}
    Full Article Content:
    --- START ARTICLE ---
    {article_content}
    --- END ARTICLE ---

    {character_mention_info}

    MISSION:
    Generate a **single** highly descriptive image generation prompt for ComfyUI (FLUX workflow).
    This prompt should capture the **overall essence, mood, or a single pivotal concept** of the *entire* news article provided above, like a striking feature photo for a magazine article. It should not be a sequence of panels, but one powerful image.

    Prompt Requirements:
    -  **Content:** Depict a scene, concept, or character(s) that best represents the core of the article. Incorporate general Ephergent universe elements like '{selected_theme}' and {', '.join(selected_elements)} where appropriate to the article's theme. If a mentioned character is central to the article's essence, depict them.
    -  **Composition:** Describe a compelling visual composition suitable for a single, impactful image.
    -  **Lighting & Mood:** Specify lighting and mood appropriate to the article's tone and dimension.
    -  **Quality:** Aim for a high quality, detailed illustration description.
    -  **Avoid:** Do not include the *main reporter* ({reporter.name}) themself in the shot unless their action/reaction is the absolute core focus of the article's essence. Do not include text overlays, panel numbers, or speech bubbles.

    OUTPUT FORMAT:
    Return ONLY a single string, which is the complete image prompt

    Generate the single prompt now.
    """
    try:
        logger.info(f"Generating single 'article essence' image prompt via Gemini...")
        response = gemini_client.generate_content(prompt_instruction)

        # Check for safety ratings and blocked prompts
        prompt_feedback = getattr(response, 'prompt_feedback', None)
        if prompt_feedback and getattr(prompt_feedback, 'block_reason', None):
             logger.error(f"Gemini prompt blocked for article essence generation. Reason: {prompt_feedback.block_reason}")
             return None
        if not response.parts and hasattr(response, 'candidates') and response.candidates:
             finish_reason = getattr(response.candidates[0], 'finish_reason', None)
             if finish_reason and finish_reason != 1: # 1 typically means "STOP" (successful)
                 logger.error(f"Gemini generation finished unexpectedly. Reason: {finish_reason}")
                 safety_ratings = getattr(response.candidates[0], 'safety_ratings', [])
                 if safety_ratings: logger.error(f"Safety Ratings: {safety_ratings}")
                 return None

        # Extract text/JSON
        if response.parts:
            response_text = response.parts[0].text
        elif hasattr(response, 'text'):
            response_text = response.text
        else:
            logger.error("Gemini response for article essence prompt missing text content.")
            logger.debug(f"Full Gemini response object: {response}")
            return None

        logger.debug(f"Raw Gemini response for essence prompt:\n{response_text}")

        # The response should be a single string, potentially wrapped in markdown fences
        cleaned_response_text = re.sub(r'^```(?:json)?\n', '', response_text.strip(), flags=re.MULTILINE)
        cleaned_response_text = re.sub(r'\n```$', '', cleaned_response_text, flags=re.MULTILINE)

        final_prompt = cleaned_response_text.strip()

        if not final_prompt.startswith(STYLE_PREFIX):
            logger.warning(f"Generated prompt did not start with the required prefix '{STYLE_PREFIX}'. Adding it.")
            final_prompt = f"{STYLE_PREFIX} {final_prompt}"

        logger.info(f"Successfully generated article essence image prompt.")
        logger.debug(f"Generated prompt: {final_prompt[:150]}...")
        return final_prompt

    except Exception as e:
        logger.error(f"Error generating article essence image prompt via LLM: {e}", exc_info=True)
        return None


def generate_story_image_prompt(
    story_data: Dict[str, Any],
    reporter: Reporter,
    image_type: str = "featured",
    image_index: int = 0 # Not used for featured images.
    ) -> str | None:
    """
    Generates a ComfyUI prompt based on story data and reporter persona for the FEATURED image.
    This function is NOT used for article images anymore.

    Args:
        story_data (Dict[str, Any]): Dictionary containing story details (title, content, etc.).
        reporter (Reporter): The reporter object associated with the story.
        image_type (str): Type of image ('featured').
        image_index (int): Not used for featured images.

    Returns:
        str | None: The generated image prompt string, or None on failure.
    """
    if image_type != "featured":
        logger.error(f"generate_story_image_prompt now only supports 'featured' image type.")
        return None

    title = story_data.get('title', 'Untitled Report')
    # location = story_data.get('location', 'Unknown Location') # Location not typically needed for featured

    # Extract universe themes/elements inspired by ephergent_universe_prompt.md
    dimension_themes = [
        "Urban Sci-Fi Prime Material", "Gothic Horror Nocturne", "Steampunk Cogsworth",
        "Cyberpunk AI Mechanica", "Ecological Sci-Fi Verdantia", "Cosmic Horror The Edge",
        "Time-Travel Mystery Chronos Reach", "Absurdist Bureaucracy", "Political Thriller",
        "Reality Stabilization", "Narrative Causality"
    ]
    universe_elements = [
        "The Ephergent HQ", "A1 AI Assistant", "CLX Crystallized Laughter", "Those Who Wait",
        "Reality Anchors", "Narrative Engine", "Dimensional Rifts", "Temporal Paradoxes",
        "Sentient Infrastructure", "Reality Glitches", "Void Incursions", "Anti-Creation",
        "Great Thought-Root Network", "Cybernetic Dinosaurs", "Gravity Reversals",
        "Multi-directional Time Flow", "Probability Storms", "Data Streams", "Holographic Interfaces",
        "Quantum Computing", "Espresso Machine Robot", "Dimensional Barriers", "Forbidden Knowledge",
        "Cyclical Collapse", "Reality Fatigue", "Rogue Narrative Loops"
    ]
    visual_keywords = [
        "unpredictable physics", "obsidian architecture", "stained-glass observatories", "clockwork mechanisms",
        "artisanal tea", "neon-lit megacities", "telepathic plants", "glowing root networks",
        "half-formed reality", "time loops", "fragmented memories", "shifting geometric patterns",
        "flowing streams of light", "ancient symbols", "star charts", "data constructs", "cosmic void",
        "glitching displays", "unstable energy fluctuations", "corrupted data", "fractal patterns",
        "impossible geometries", "patchwork realities", "shadowy figures", "metallic tang of ozone",
        "cascading waterfalls (up and down)", "high-contrast lighting", "flickering neon", "shadowy ambiance"
    ]

    # Select relevant theme/elements based on reporter/story context if possible
    selected_theme = random.choice(reporter.topics) if reporter.topics else random.choice(dimension_themes)
    # Ensure selected_theme is a string
    if not isinstance(selected_theme, str):
        selected_theme = random.choice(dimension_themes) # Fallback

    num_elements = random.randint(1, 3)
    selected_elements = random.sample(universe_elements + visual_keywords, k=num_elements)

    # --- Featured Image Prompt Generation Logic ---
    side_of_text = random.choice(["on the right. On the left side", "on the left. On the right side"])
    font_color_combo = random.choice([
        "light blue bold text with white", "dark red bold text with white", "white bold text with blue",
        "dark blue bold text with white", "dark green bold text with white",
        "light green bold text with white", "light red bold text with white", "purple bold text with white",
        "yellow bold text with black", "orange bold text with black", "neon pink bold text with black"
    ])
    # Clean title for display (take first part, uppercase, remove extra chars)
    title_first_part = re.sub(r'[^\w\s]', '', title.split(':')[0]).strip().upper()
    words = title_first_part.split()
    display_title = ' '.join(words[:2]) # Limit to first 2 words

    # Use reporter's SD prompt fragment if available, otherwise generate description
    if reporter.stable_diffusion_prompt:
        character_desc = reporter.stable_diffusion_prompt # Use pre-defined description
    else:
        # Fallback: generate a simple description based on name/role
        character_desc = f"the reporter {reporter.name}, interdimensional correspondent for The Ephergent"

    prompt_text = f"""
    {STYLE_PREFIX} {font_color_combo} hard drop shadow that
    reads '{display_title}' {side_of_text} of image there is {character_desc},
    incorporating elements of '{selected_theme}' and {', '.join(selected_elements)}, black background. High detail, evocative.
    """
    logger.info(f"Generated 'featured' image prompt: {prompt_text}")

    # Clean up prompt text
    cleaned_prompt = " ".join([line.strip() for line in prompt_text.strip().splitlines() if line.strip()])
    return cleaned_prompt


# --- ComfyUI API Functions ---

def open_websocket_connection() -> tuple[websocket.WebSocket, str] | tuple[None, None]:
    """Open a websocket connection to ComfyUI"""
    client_id = str(uuid.uuid4())
    ws_url = f"ws://{COMFYUI_URL.replace('http://', '').replace('https://', '')}/ws?clientId={client_id}"
    try:
        ws = websocket.WebSocket()
        ws.connect(ws_url)
        logger.info(f"WebSocket connection established: {ws_url}")
        return ws, client_id
    except (websocket.WebSocketException, ConnectionRefusedError, TimeoutError) as e:
        logger.error(f"Failed to connect to ComfyUI WebSocket at {ws_url}: {e}")
        logger.error("Please ensure ComfyUI is running and accessible.")
        return None, None
    except Exception as e:
        logger.error(f"Unexpected error connecting to WebSocket at {ws_url}: {e}")
        return None, None


def queue_prompt(prompt_workflow: dict, client_id: str) -> str | None:
    """Queue a prompt workflow for processing by ComfyUI"""
    if not COMFYUI_URL:
        logger.error("COMFYUI_URL is not set. Cannot queue prompt.")
        return None

    p = {"prompt": prompt_workflow, "client_id": client_id}
    data = json.dumps(p).encode('utf-8')
    url = f"{COMFYUI_URL}/prompt"
    headers = {'Content-Type': 'application/json', 'Accept': 'application/json'}
    try:
        response = requests.post(url, data=data, headers=headers, timeout=30) # Added timeout
        response.raise_for_status()
        response_data = response.json()
        if 'prompt_id' not in response_data:
            logger.error(f"Unexpected response from ComfyUI queue: {response_data}")
            return None
        logger.info(f"Queued prompt with ID: {response_data['prompt_id']}")
        return response_data['prompt_id']
    except requests.exceptions.Timeout:
        logger.error(f"Timeout connecting to ComfyUI at {url}")
        return None
    except requests.exceptions.ConnectionError:
        logger.error(f"Connection error connecting to ComfyUI at {url}. Is it running?")
        return None
    except requests.RequestException as e:
        logger.error(f"Failed to queue prompt: {e}")
        if e.response is not None:
            logger.error(f"Response status: {e.response.status_code}")
            logger.error(f"Response text: {e.response.text[:500]}") # Limit response text logging
        return None
    except Exception as e:
        logger.error(f"Unexpected error queuing prompt: {e}")
        return None


def get_image(filename: str, subfolder: str, folder_type: str) -> bytes | None:
    """Get an image from ComfyUI"""
    if not COMFYUI_URL:
        logger.error("COMFYUI_URL is not set. Cannot get image.")
        return None
    params = {"filename": filename, "subfolder": subfolder, "type": folder_type}
    url = f"{COMFYUI_URL}/view?{urlencode(params)}"
    logger.info(f"Attempting to download image from: {url}") # Added log
    try:
        response = requests.get(url, timeout=120) # Increased timeout
        response.raise_for_status()
        logger.info(f"Successfully downloaded image '{filename}'")
        return response.content
    except requests.exceptions.Timeout:
        logger.error(f"Timeout fetching image {filename} from {url}")
        return None
    except requests.exceptions.ConnectionError:
        logger.error(f"Connection error fetching image from {url}.")
        return None
    except requests.RequestException as e:
        logger.error(f"Failed to fetch image {filename}: {e}")
        if e.response is not None:
            logger.error(f"Response status: {e.response.status_code}")
            logger.error(f"Response content: {e.response.content[:200].decode('utf-8', errors='ignore')}") # Limit response text
        return None
    except Exception as e:
        logger.error(f"Unexpected error fetching image {filename}: {e}")
        return None


def get_history(prompt_id: str) -> dict | None:
    """Get the execution history for a prompt"""
    if not COMFYUI_URL:
        logger.error("COMFYUI_URL is not set. Cannot get history.")
        return None
    url = f"{COMFYUI_URL}/history/{prompt_id}"
    try:
        response = requests.get(url, timeout=30) # Added timeout
        response.raise_for_status()
        return response.json()
    except requests.exceptions.Timeout:
        logger.error(f"Timeout fetching history for {prompt_id} from {url}")
        return None
    except requests.exceptions.ConnectionError:
        logger.error(f"Connection error fetching history from {url}.")
        return None
    except requests.RequestException as e:
        logger.error(f"Failed to fetch history for {prompt_id}: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error fetching history for {prompt_id}: {e}")
        return None


def track_progress(prompt_workflow: dict, ws: websocket.WebSocket, prompt_id: str, client_id: str):
    """Track the progress of a prompt execution via WebSocket"""
    logger.debug(f"Tracking progress for prompt_id: {prompt_id} with client_id: {client_id}")
    nodes_total = len(prompt_workflow)
    nodes_executed = 0
    last_node_id = None # Track the last node that finished execution

    # Identify the save node ID(s) heuristically
    save_node_ids = {
        nid for nid, node_info in prompt_workflow.items()
        if node_info.get("class_type", "").startswith("SaveImage")
    }
    if not save_node_ids:
        logger.warning("Could not identify a SaveImage node in the workflow. Progress tracking might stop early.")
    else:
        logger.info(f"Identified potential SaveImage node(s): {save_node_ids}")

    while True:
        try:
            out = ws.recv() # This blocks until a message is received
            if isinstance(out, str):
                message = json.loads(out)
                msg_type = message.get('type')
                data = message.get('data', {})

                if msg_type == 'status':
                    sid = data.get('sid')
                    queue_remaining = data.get('status', {}).get('exec_info', {}).get('queue_remaining', 0)
                    if sid == client_id:
                         logger.info(f"Queue status update: {queue_remaining} item(s) remaining.")
                         # We could potentially exit early if queue_remaining becomes 0 *after* our job started,
                         # but relying on 'executed' message for the save node is more robust.

                elif msg_type == 'progress':
                    progress_data = message['data']
                    logger.info(f"Node {progress_data.get('node', '?')} Progress: Step {progress_data['value']} of {progress_data['max']}")

                elif msg_type == 'executing':
                     node_id = data.get('node')
                     prompt_id_msg = data.get('prompt_id')
                     if node_id and prompt_id_msg == prompt_id:
                         logger.info(f"Executing node {node_id}...")

                elif msg_type == 'executed':
                    executed_data = message['data']
                    exec_prompt_id = executed_data.get('prompt_id')
                    node_id = executed_data.get('node')

                    if exec_prompt_id == prompt_id and node_id:
                        nodes_executed += 1
                        last_node_id = node_id # Update last executed node
                        logger.info(f"Node {node_id} executed ({nodes_executed}/{nodes_total}).")
                        # Check if this executed node is one of the save nodes
                        if node_id in save_node_ids:
                             logger.info(f"Save node {node_id} executed. Assuming prompt {prompt_id} is complete.")
                             return # Exit tracking loop

            elif isinstance(out, bytes):
                logger.debug("Received binary WebSocket message.")
                # Handle binary if necessary

        except websocket.WebSocketConnectionClosedException:
            logger.warning("WebSocket connection closed during progress tracking.")
            break # Exit loop
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode WebSocket message: {out} - Error: {e}")
            # Continue listening if possible, but log the error
        except Exception as e:
            logger.error(f"Error during progress tracking: {e}", exc_info=True)
            break # Exit loop on other errors

    # Fallback exit condition if save node wasn't detected or message missed
    logger.warning(f"Exiting progress tracking for {prompt_id}. Last executed node: {last_node_id}. Check history if image is missing.")


def get_images_from_history(prompt_id: str) -> list[bytes]:
    """Extracts final output image data blobs from the execution history."""
    history_data = get_history(prompt_id)
    if not history_data or prompt_id not in history_data:
        logger.warning(f"History not found or incomplete for prompt ID: {prompt_id}")
        return []

    prompt_history = history_data[prompt_id]
    output_images_data = []

    if 'outputs' in prompt_history:
        logger.debug(f"Processing outputs for prompt {prompt_id}. Found nodes: {list(prompt_history['outputs'].keys())}")
        # Iterate through nodes in history outputs
        for node_id, node_output in prompt_history['outputs'].items():
            # Check if this node output contains images
            if 'images' in node_output:
                logger.debug(f"Found 'images' key in output of node {node_id}")
                for image_info in node_output['images']:
                    image_type = image_info.get('type')
                    filename = image_info.get('filename')
                    subfolder = image_info.get('subfolder', '')
                    logger.debug(f"  - Found image: filename='{filename}', type='{image_type}', subfolder='{subfolder}'")
                    # We are typically interested in the final 'output' type images saved by SaveImage nodes
                    # 'temp' images might exist from previews, but we usually want the saved file.
                    if image_type == 'output':
                        if filename:
                            logger.info(f"Attempting to retrieve final output image: node={node_id}, filename='{filename}', subfolder='{subfolder}'")
                            image_data = get_image(filename, subfolder, image_type)
                            if image_data:
                                output_images_data.append(image_data)
                            else:
                                logger.warning(f"Failed to retrieve image data for '{filename}' from node {node_id}")
                        else:
                            logger.warning(f"Image entry with type 'output' found in history for node {node_id} but missing filename.")
                    else:
                        logger.debug(f"Skipping non-output image type '{image_type}' in node {node_id}")
            else:
                 logger.debug(f"No 'images' key in output of node {node_id}") # Reduce log noise
    else:
        logger.warning(f"No 'outputs' key found in history for prompt ID: {prompt_id}")

    if not output_images_data:
         logger.warning(f"No final output images successfully retrieved from history for prompt ID: {prompt_id}")

    return output_images_data


def generate_image_with_comfyui(
        prompt: str,
        workflow_filepath: Path,
        output_dir: Path,
        filename: str,
        change_resolution: bool = False) -> Path | None:
    """
    Generates an image using the ComfyUI API, saves it, and returns the path.

    Args:
        prompt (str): The text prompt for image generation.
        workflow_filepath (Path): Path to the ComfyUI workflow JSON file.
        output_dir (Path): Directory where the final image should be saved.
        filename (str): The desired filename for the output image (e.g., 'featured.png').
        change_resolution (bool): Whether to change the resolution to 1024x1024.

    Returns:
        Path | None: The path to the saved image file, or None on failure.
    """
    output_path = output_dir / filename
    output_dir.mkdir(parents=True, exist_ok=True) # Ensure output directory exists

    # --- Load Workflow ---
    try:
        logger.info(f"Loading workflow from: {workflow_filepath}")
        with open(workflow_filepath, 'r', encoding='utf-8') as f:
            workflow = json.load(f)
    except FileNotFoundError:
        logger.error(f"Workflow file not found: {workflow_filepath}")
        return None
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON in workflow file: {workflow_filepath}")
        return None
    except Exception as e:
        logger.error(f"Error loading workflow {workflow_filepath}: {e}")
        return None

    # --- Connect WebSocket ---
    ws, client_id = open_websocket_connection()
    if not ws:
        return None # Connection failed

    try:
        # --- Modify Workflow ---
        # 1. Update Prompt Node(s)
        prompt_node_updated = False
        # Heuristic: Find nodes named "PROMPT" or CLIPTextEncode connected to KSampler positive input
        # This might need adjustment based on the specific workflow structure
        positive_prompt_node_ids = set()
        negative_prompt_node_ids = set() # Track negative too

        # Find KSampler nodes and their inputs
        ksampler_nodes = {nid: ndata for nid, ndata in workflow.items() if "KSampler" in ndata.get("class_type", "")}

        for ksampler_id, ksampler_data in ksampler_nodes.items():
            inputs = ksampler_data.get("inputs", {})
            pos_input = inputs.get("positive")
            neg_input = inputs.get("negative")
            if isinstance(pos_input, list) and len(pos_input) > 0:
                positive_prompt_node_ids.add(pos_input[0]) # Assumes [node_id, output_index]
            if isinstance(neg_input, list) and len(neg_input) > 0:
                negative_prompt_node_ids.add(neg_input[0])

        if not positive_prompt_node_ids:
             logger.warning("Could not identify positive prompt node linked to KSampler. Trying common names (e.g., 'PROMPT', CLIPTextEncode).")
             # Fallback: Look for nodes with specific titles or class types
             for node_id, node_data in workflow.items():
                 meta_title = node_data.get('_meta', {}).get('title', '').upper()
                 class_type = node_data.get('class_type', '')
                 # Check for title 'PROMPT' or class type containing 'CLIPTextEncode'
                 if meta_title == 'PROMPT' or "CLIPTextEncode" in class_type:
                     # Avoid adding known negative prompts here
                     if node_id not in negative_prompt_node_ids:
                         positive_prompt_node_ids.add(node_id)


        if not positive_prompt_node_ids:
             logger.error("Could not find any potential positive prompt node (e.g., CLIPTextEncode) in workflow.")
             logger.warning("Proceeding without updating prompt node. Image may not match request.") # Option 2: Warn and continue

        for node_id in positive_prompt_node_ids:
             if node_id in workflow and 'inputs' in workflow[node_id]:
                 workflow[node_id]['inputs']['text'] = prompt
                 logger.info(f"Updated positive prompt node {node_id} with new prompt.")
                 prompt_node_updated = True
             else:
                 logger.warning(f"Identified node {node_id} as potential prompt node, but couldn't find/update it in workflow.")

        # 2. Update Seed in KSampler Node(s)
        # Get seed from .env or generate a random one
        if os.getenv('COMFYUI_SEED'):
            seed = int(os.getenv('COMFYUI_SEED'))
            logger.info(f"Using seed from .env: {seed}")
        else:
            seed = random.randint(10**14, 10**15 - 1)
        ksampler_updated = False
        for node_id in ksampler_nodes: # Iterate only over KSampler nodes found earlier
            if 'inputs' in workflow[node_id]:
                workflow[node_id]['inputs']['seed'] = seed
                logger.info(f"Updated KSampler node {node_id} with seed: {seed}")
                ksampler_updated = True
            else:
                logger.warning(f"Found KSampler node {node_id} but it has no 'inputs'.")

        if not ksampler_updated:
             logger.warning("Could not find KSampler node in workflow to update seed. Using default seed.")


        # 3 Change resolution to 1024x1024
        # Check if user wants to change resolution 'change_resolution'
        if change_resolution:
            logger.info("Changing resolution to 1024x1024.")
            # Heuristic: Find the 'EmptyLatentImage' node
            resolution_node_id = None
            for node_id, node_data in workflow.items():
                if node_data.get("class_type") == "EmptyLatentImage":
                    resolution_node_id = node_id
                    break # Assume only one such node

            if resolution_node_id and 'inputs' in workflow[resolution_node_id]:
                workflow[resolution_node_id]['inputs']['width'] = 1024
                workflow[resolution_node_id]['inputs']['height'] = 1024
                logger.info(f"Updated resolution to 1024x1024 in node {resolution_node_id}.")
            else:
                logger.warning(f"Could not find EmptyLatentImage node or its inputs to update resolution.")
        else:
            logger.info("Resolution change skipped. Keeping original resolution.")

        # 3. Update Output Filename in SaveImage Node(s)
        # save_node_updated = False
        # output_filename_stem = Path(filename).stem # Get filename without extension
        # for node_id, node_data in workflow.items():
        #      if node_data.get("class_type", "").startswith("SaveImage"):
        #          if 'inputs' in workflow[node_id]:
        #              workflow[node_id]['inputs']['filename_prefix'] = output_filename_stem
        #              # Ensure foldername_prefix is empty if it exists, to avoid unwanted subdirs
        #              if 'foldername_prefix' in workflow[node_id]['inputs']:
        #                  workflow[node_id]['inputs']['foldername_prefix'] = ""
        #                  logger.info(f"Ensured foldername_prefix is empty in SaveImage node {node_id}.")
        #              logger.info(f"Updated SaveImage node {node_id} filename_prefix to: {output_filename_stem}")
        #              save_node_updated = True
        #          else:
        #              logger.warning(f"Found SaveImage node {node_id} but it has no 'inputs'.")
        #          # break # Assuming one SaveImage? Remove break if multiple need updating
        #
        # if not save_node_updated:
        #      logger.warning("Could not find SaveImage node in workflow to update filename. Image might save with default name.")

        # --- Queue and Track ---
        prompt_id = queue_prompt(workflow, client_id)
        if not prompt_id:
            logger.error("Failed to queue prompt with ComfyUI.")
            return None

        track_progress(workflow, ws, prompt_id, client_id)

        # --- Retrieve Image ---
        logger.info(f"Attempting to retrieve images from history for prompt_id: {prompt_id}")
        # Add a small delay to allow history/file system to potentially update after 'executed' signal
        logger.info("Waiting longer before fetching history...")
        # time.sleep(30) # Increased delay
        images_data = get_images_from_history(prompt_id)

        if images_data:
            # Save the first image retrieved
            try:
                logger.info(f"Saving image data to: {output_path}")
                with open(output_path, 'wb') as f:
                    f.write(images_data[0])
                logger.info(f"Image successfully generated and saved to: {output_path}")
                if output_path.exists():
                    logger.info(f"Confirmed image file exists at: {output_path}")
                    return output_path
                else:
                    logger.error(f"Image file not found after save: {output_path}")
                    return None
            except IOError as e:
                logger.error(f"Failed to save image to {output_path}: {e}")
                return None
            except Exception as e:
                logger.error(f"Unexpected error saving image {output_path}: {e}")
                return None
        else:
            logger.error(f"No images found in history for prompt ID: {prompt_id}. Image generation likely failed or history is delayed/incomplete.")
            # Optional: Add fallback check to ComfyUI output folder if needed, but history is preferred.
            return None

    except Exception as e:
        logger.error(f"An unexpected error occurred during ComfyUI image generation: {e}", exc_info=True)
        return None
    finally:
        if ws and ws.connected:
            logger.info("Closing WebSocket connection.")
            ws.close()


# --- Main Orchestration Function ---

def generate_ephergent_image(
    story_data: Dict[str, Any], # Use story data instead of separate args
    reporter: Reporter,
    image_type: str, # "featured" or "article"
    output_dir: Path,
    filename: str,
    workflow_path: Path = DEFAULT_WORKFLOW_FILE,
    image_index: int = 0, # Index for article images (used for logging/tracking)
    prompt_override: Optional[str] = None, # Add override parameter
    prompt_details_list: Optional[List[Dict[str, Any]]] = None # Add list to store prompt info
    ) -> Path | None:
    """
    Generates an image using ComfyUI. For 'featured' images, it generates the prompt.
    For 'article' images, it expects a `prompt_override`. Optionally stores prompt details.

    Args:
        story_data: Dictionary containing story details (title, content, etc.).
        reporter: The Reporter object.
        image_type: "featured" or "article".
        output_dir: Directory to save the image.
        filename: Desired filename for the image.
        workflow_path: Path to the ComfyUI workflow JSON.
        image_index: Index for article images (used for logging/tracking).
        prompt_override: If provided, use this prompt directly instead of generating one.
                         Required for image_type="article".
        prompt_details_list: Optional list to append prompt details dictionary to.

    Returns:
        Path | None: Path to the generated image or None on failure.
    """
    logger.info(f"--- Starting {image_type} image generation (Index: {image_index}) ---")
    logger.info(f"Reporter: {reporter.name}, Title: {story_data.get('title', 'N/A')[:50]}...")

    final_prompt_text = None

    if prompt_override:
        logger.info(f"Using provided prompt override for {image_type} image.")
        final_prompt_text = prompt_override
    elif image_type == "featured":
        logger.info("Generating prompt for featured image...")
        final_prompt_text = generate_story_image_prompt(
            story_data=story_data,
            reporter=reporter,
            image_type="featured"
            # image_index is ignored for featured
        )
    else:
        # This case should ideally not happen if called correctly from app.py
        logger.error(f"Prompt override is required for image_type='{image_type}', but none was provided.")
        return None

    if not final_prompt_text:
        logger.error(f"Failed to obtain image prompt for {image_type} image.")
        return None

    # Store prompt details if list is provided
    # This should happen *before* generation, using the final_prompt_text
    if prompt_details_list is not None:
        prompt_info = {
            "image_type": image_type,
            "filename": filename,
            "prompt": final_prompt_text,  # Store the actual prompt used
        }
        if image_type == "article":
            prompt_info["index"] = image_index  # Keep index for potential future use or logging

        # Avoid adding duplicates if somehow called multiple times for the same image
        if not any(p.get("filename") == filename for p in prompt_details_list):
            prompt_details_list.append(prompt_info)
            logger.info(f"Stored prompt details for {filename}")
        else:
            logger.debug(f"Prompt details for {filename} already exist. Skipping storage.")


    # 2. Generate Image using ComfyUI

    generated_image_path = generate_image_with_comfyui(
        prompt=final_prompt_text, # Use the determined prompt
        workflow_filepath=workflow_path,
        output_dir=output_dir,
        filename=filename, # Pass the full desired filename (e.g., "featured.png" or "article_essence.png")
        change_resolution=False # Apply resolution change for article essence image
    )

    if generated_image_path:
        logger.info(f"--- Successfully generated {image_type} image: {generated_image_path} ---")
        return generated_image_path
    else:
        logger.error(f"--- Failed to generate {image_type} image ---")
        # Remove the corresponding prompt detail if generation failed? Optional.
        # if prompt_details_list is not None:
        #     # Find and remove the entry matching the failed filename
        #     prompt_details_list[:] = [d for d in prompt_details_list if d.get("filename") != filename]
        return None


# Example usage when run directly (Updated for new structure)
if __name__ == "__main__":
    print("--- Running Image Generation Test (Simulating Story Processing) ---")

    # Check prerequisites
    if not GEMINI_API_KEY:
        print("ERROR: GEMINI_API_KEY not found in environment variables. Cannot generate prompts. Skipping test.")
        exit()
    if not os.getenv('COMFYUI_URL'):
        print("ERROR: COMFYUI_URL not found in environment variables. Cannot connect to ComfyUI. Skipping test.")
        exit()
    if not DEFAULT_WORKFLOW_FILE.exists():
        print(f"ERROR: Default workflow file not found at {DEFAULT_WORKFLOW_FILE}. Skipping test.")
        exit()

    try:
        # 1. Setup Output Directory
        test_output_dir = Path(__file__).parent.parent / 'output' / 'image_generator_test'
        test_output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Using output directory: {test_output_dir}")

        # 2. Select Reporter
        print("\nSelecting a reporter (pixel_paradox)...")
        test_reporter = Reporter(identifier="pixel_paradox") # Use a specific reporter
        if not test_reporter or not test_reporter.reporter_data:
             print("ERROR: Failed to get reporter 'pixel_paradox'. Exiting.")
             exit()
        print(f"Using Reporter: {test_reporter.name} ({test_reporter.id})")

        # 3. Dummy Story Data (replace with actual data if needed)
        test_story = {
            "title": "REALITY FAULT LINES EMERGE ACROSS PRIME MATERIAL",
            "Filed by": "Pixel Paradox, Interdimensional Correspondent",
            "location": "Sector 7 --- Fractal Storm Fields",
            "stardate": "77.4 Glitch Standard",
            "content": "A pounding headache and the metallic tang of ozone. That's how I arrived in Sector 7 today. One moment, I'm calibrating my reality-encryption eyewear, the next, I'm dodging rogue lightning strikes in a storm that's glitching through a dozen dimensions at once. Reality is getting choppy out here, folks...\nWe've got a Grade-A convergence crisis... Objects, energy, even chunks of landscape from other dimensions are bleeding into Prime Material... mathematical pattern... someone messing with the equation.\nI managed to link up with Luminara amidst the chaos... waterfall cascading both up and down... five different dimensional signatures...\nThe sky fractured, revealing glimpses of impossible geometries... panicked energy readings.\nSomething big is coming... Stay weird..."
        }
        print(f"Using Story Title: {test_story['title']}")

        # 4. Prepare Filenames
        sanitized_title = sanitize_filename(test_story['title'])
        filename_base = f"test_{test_reporter.id}_{sanitized_title[:50]}" # Limit length

        # List to store prompt details
        test_prompt_details = []

        # --- Test Feature Image ---
        print("\n--- Testing FEATURED Image ---")
        feature_filename = f"{filename_base}_featured.png"
        feature_image_path = generate_ephergent_image(
            story_data=test_story,
            reporter=test_reporter,
            image_type="featured",
            output_dir=test_output_dir,
            filename=feature_filename,
            workflow_path=DEFAULT_WORKFLOW_FILE, # Explicitly pass default
            prompt_details_list=test_prompt_details # Pass the list
            # prompt_override is None for featured
        )
        if feature_image_path:
            print(f"Featured image generated: {feature_image_path}")
        else:
            print("Featured image generation failed.")

        # --- Test Article Essence Image (Single) ---
        print(f"\n--- Testing single ARTICLE Essence Image ---")

        # Generate the single prompt first
        article_essence_prompt = generate_article_essence_image_prompt(
            story_data=test_story,
            reporter=test_reporter
        )

        article_image_path = None # Will hold the single path
        if not article_essence_prompt:
            print("ERROR: Failed to generate article essence prompt. Skipping article image generation.")
        else:
            # Store the generated prompt *before* generating the image
            article_filename = f"{filename_base}_article_essence.png" # Use a specific name
            prompt_info = {
                "image_type": "article",
                "filename": article_filename,
                "prompt": article_essence_prompt, # Store the actual prompt used
                "index": 0 # Index 0 for the single image
            }
            # Avoid adding duplicates if somehow called multiple times for the same image
            if not any(p.get("filename") == article_filename for p in test_prompt_details):
                test_prompt_details.append(prompt_info)
                logger.info(f"Stored pre-generated prompt details for {article_filename}")
            else:
                logger.debug(f"Prompt details for {article_filename} already exist. Skipping storage.")

            # Now generate the image using the pre-generated prompt
            print(f"\nGenerating Article Essence Image using pre-generated prompt...")
            article_image_path = generate_ephergent_image(
                story_data=test_story, # Still needed for context/logging inside function
                reporter=test_reporter, # Still needed for context/logging inside function
                image_type="article",
                output_dir=test_output_dir,
                filename=article_filename,
                image_index=0, # Index 0 for the single image
                prompt_override=article_essence_prompt, # Pass the specific prompt
                prompt_details_list=test_prompt_details # Pass list (already populated)
            )
            if article_image_path:
                print(f"Article essence image generated: {article_image_path}")
            else:
                print(f"Article essence image generation failed.")


        # Save the collected prompt details
        prompts_json_path = test_output_dir / f"{filename_base}_image_prompts.json"
        try:
            # Ensure no duplicates before saving (shouldn't happen with new logic, but safe)
            seen_filenames = set()
            unique_prompts = []
            for p in test_prompt_details:
                fname = p.get("filename")
                if fname not in seen_filenames:
                    unique_prompts.append(p)
                    seen_filenames.add(fname)

            with open(prompts_json_path, 'w', encoding='utf-8') as f:
                json.dump(unique_prompts, f, indent=2)
            print(f"\nSaved image prompt details to: {prompts_json_path}")
        except Exception as e:
            print(f"\nError saving prompt details: {e}")


        print("\n--- Image Generation Test Complete ---")

    except ImportError as e:
        print(f"\nERROR: Missing import for testing - {e}. Please ensure all utils are available.")
    except Exception as e:
        print(f"\nAn unexpected error occurred during the test: {e}")
        logger.error("Test run failed.", exc_info=True)

