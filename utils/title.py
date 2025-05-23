#!/usr/bin/env python3
import os
import logging
import re
import time
import json
from pathlib import Path
from dotenv import load_dotenv

# Try to import genai, handle import error if needed
try:
    import google.generativeai as genai
except ImportError:
    print("Error: google-generativeai package not found. Please install it: pip install google-generativeai")
    genai = None # Set to None if import fails

# Local imports
from utils.reporter import Reporter
from utils.system_prompt import load_system_prompt # Import the refactored function

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# --- Gemini Client Initialization ---
GEMINI_MODEL = os.getenv('GEMINI_MODEL', 'gemini-2.0-flash')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
gemini_client = None
if genai and GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_client = genai.GenerativeModel(GEMINI_MODEL)
        logger.info(f"Gemini client initialized successfully with {GEMINI_MODEL}.")
    except Exception as e:
        logger.error(f"Failed to initialize Gemini client for summarization: {e}")
elif not genai:
    logger.error("Gemini client cannot be initialized for summarization (package missing).")
elif not GEMINI_API_KEY:
    logger.error("Gemini client cannot be initialized for summarization (API key missing).")


def generate_titles(reporter: Reporter, topic: str, article_content: str, n_titles: int = 5) -> list[str]:
    """
    Generates a list of potential article titles based on the topic, content, and reporter's persona.

    Args:
        reporter (Reporter): An initialized Reporter object containing the persona.
        topic (str): The central topic of the article.
        article_content (str): The full text content of the generated article.
        n_titles (int): The number of titles to generate.

    Returns:
        list[str]: A list of generated titles, or an empty list if generation failed.
    """
    if not gemini_client:
        logger.error("Gemini client is not available. Cannot generate titles.")
        return []
    if not reporter or not reporter.prompt:
        logger.error("Invalid reporter or missing reporter prompt. Cannot generate titles.")
        return []
    if not article_content:
        logger.warning("Article content is empty. Title generation might lack context.")
        # Proceed, but the quality might be lower

    # Get the Ephergent system prompt
    system_prompt = load_system_prompt()

    # Construct the prompt for Gemini
    prompt = f"""
    {system_prompt}
    MISSION:
    {reporter.prompt}
    
    You are {reporter.name}, a headline writer for "The Ephergent". 
    Based on the following article topic and content, generate {n_titles} eye-catching, imaginative, and absurd headline titles.
    
    Article Topic: {topic}
    
    Article Content:
    ---
    {article_content}
    ---
    
    Headline Requirements:
    - Generate exactly {n_titles} distinct headlines.
    - Each headline should be provocative and attention-grabbing.
    - Incorporate terms, concepts, or the general tone from The Ephergent universe described in your persona.
    - Match your specific reporting style and voice (e.g., punky, analytical, glitchy, enigmatic, flamboyant, avant-garde).
    - Headlines should be relatively concise, ideally between 5-15 words.
    - Often include a surprising twist, juxtaposition, or absurd element related to the article content.
    - May use made-up slang or terms appropriate to your persona.
    - Format the output as a JSON list of strings. Example: ["Headline 1", "Headline 2", "Headline 3"]
    
    Output ONLY the JSON list of strings. Do not include any preamble or explanation.
    """

    logger.info(f"Generating {n_titles} titles for topic: '{topic}' using reporter {reporter.name}")
    logger.debug(f"Title prompt sent to Gemini:\n--- START PROMPT ---\n{prompt[:200]}...\n--- END PROMPT ---")
    start_time = time.time()

    try:
        # Generate content using Gemini API
        response = gemini_client.generate_content(prompt)
        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.info(f"Title generation API call completed in {elapsed_time:.2f} seconds")

        # Extract and clean the generated text
        if response.parts:
            raw_content = response.parts[0].text
        elif hasattr(response, 'text'):
             raw_content = response.text
        else:
             logger.error("Gemini response did not contain expected text parts for titles.")
             logger.debug(f"Full Gemini response for titles: {response}")
             return []

        # Attempt to parse the JSON list
        # TODO Use structured data from the model
        try:
            # Clean potential markdown fences around the JSON
            json_match = re.search(r'```(?:json)?\s*(\[.*?\])\s*```', raw_content, re.DOTALL | re.IGNORECASE)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Assume the response might be the JSON list directly, possibly with surrounding text
                start_idx = raw_content.find('[')
                end_idx = raw_content.rfind(']') + 1
                if start_idx != -1 and end_idx != 0 and end_idx > start_idx:
                    json_str = raw_content[start_idx:end_idx]
                else:
                    json_str = raw_content # Fallback if no brackets found

            titles = json.loads(json_str)
            if isinstance(titles, list) and all(isinstance(t, str) for t in titles):
                logger.info(f"Successfully parsed {len(titles)} titles from JSON response.")
                # Strip potential leading/trailing whitespace from each title
                titles = [t.strip() for t in titles]
                return titles
            else:
                logger.warning(f"Parsed JSON is not a list of strings: {titles}")
                # Fall through to manual extraction

        except json.JSONDecodeError as json_err:
            logger.warning(f"Failed to parse JSON response for titles: {json_err}. Attempting manual extraction.")
            # Fallback: Try to extract lines that look like titles
            lines = raw_content.strip().split('\n')
            potential_titles = []
            for line in lines:
                # Remove common list markers, quotes, etc.
                cleaned_line = re.sub(r'^\s*[-*]?\s*[\d.]*\s*["\']?(.*?)[",]?\s*$', r'\1', line).strip()
                # Basic check for title-like lines (avoid empty lines or instructions)
                if cleaned_line and len(cleaned_line) > 5 and not cleaned_line.lower().startswith("here are") and not cleaned_line.lower().startswith("sure,"):
                    potential_titles.append(cleaned_line)

            if potential_titles:
                 logger.info(f"Manually extracted {len(potential_titles)} potential titles.")
                 return potential_titles[:n_titles] # Return up to the requested number
            else:
                 logger.error("Could not extract titles manually from response.")
                 logger.debug(f"Raw response content: {raw_content}")
                 return []

    except Exception as e:
        logger.error(f"Error generating titles with Gemini: {e}")
        logger.debug(f"Reporter: {reporter.name}, Topic: {topic}", exc_info=True)
        return []

# Example usage when run directly
if __name__ == "__main__":
    print("Testing Title Generation...")

    if not GEMINI_API_KEY:
        print("GEMINI_API_KEY not found in environment variables. Skipping test.")
    else:
        try:
            # Use default reporter or specify one
            test_reporter = Reporter()
            print(f"Using Reporter: {test_reporter.name}")

            # Example article content (use the same as summarize test)
            example_article = """
            Alright dimension-hoppers, Pixel Paradox here, jacking straight into the hyper-cortex of the latest market madness! Forget your boring Earth-bound stocks, we're talking Crystallized Laughter Futures on The Sizzle dimension. Yeah, you heard me. Turns out bottling pure joy is big business, especially when Probability Zero keeps messing with the supply chain.

            Sources – let's call 'em 'Sparky' from Frequencia and a very shadowy figure known only as 'The Broker' from the Umbral Plane – tell me volatility is off the charts. One minute, a giggle-gram is worth more than a cyber-dino's pension fund, the next it's cheaper than recycled temporal anomalies. "It's pure chaos, Pixel," Sparky resonated, his voice causing my fillings to vibrate sympathetically. "One misplaced chuckle in Temporalius and the whole market tanks!"

            The Broker, communicating via modulated darkness, was more blunt: "Profit margins are illusory. Invest in existential dread futures instead. More stable." Typical Umbral Plane pessimism. But maybe they've got a point? The Cloud Parliament in Sector 7 is threatening regulations, citing 'unstable emotional derivatives,' and Verdantia's telepathic houseplants are whispering about a market corner attempt.

            So, what's the play? Honestly, your guess is as good as mine. Maybe short the whole concept of happiness? Or just stick to trading black market hypermaterials? That's the kind of grax-level nonsense only a timeline tourist would try to predict. Stay weird and keep your phase-shifters calibrated, folks. This ride ain't over.
            """
            example_topic = "the fluctuating economy of crystallized laughter"
            print(f"\nUsing Topic: {example_topic}")
            print(f"Using Example Article (first 100 chars): {example_article[:100]}...")

            num_titles_to_generate = 5
            generated_titles = generate_titles(test_reporter, example_topic, example_article, n_titles=num_titles_to_generate)

            if generated_titles:
                print(f"\n--- Generated Titles ({len(generated_titles)}) ---")
                for i, title in enumerate(generated_titles, 1):
                    print(f"{i}. {title}")
                print("--- End of Titles ---")
            else:
                print("\nTitle generation failed.")

        except Exception as e:
            print(f"\nAn error occurred during the test: {e}")
