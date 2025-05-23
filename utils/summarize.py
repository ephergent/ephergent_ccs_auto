#!/usr/bin/env python3
import os
import logging
import re
import time
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


def generate_summary(reporter: Reporter, article_content: str, word_count: int = 50) -> str | None:
    """
    Generates a concise summary of an article in the reporter's voice using the Gemini API.

    Args:
        reporter (Reporter): An initialized Reporter object containing the persona.
        article_content (str): The full text content of the article to summarize.
        word_count (int): Approximate desired word count for the summary.

    Returns:
        str | None: The generated summary as a string, or None if generation failed.
    """
    if not gemini_client:
        logger.error("Gemini client is not available. Cannot generate summary.")
        return None
    if not reporter or not reporter.prompt:
        logger.error("Invalid reporter or missing reporter prompt. Cannot generate summary.")
        return None
    if not article_content:
        logger.error("Article content is empty. Cannot generate summary.")
        return None

    # Get the Ephergent system prompt
    system_prompt = load_system_prompt()

    # Construct the prompt for Gemini
    prompt = f"""
    {system_prompt}
    MISSION:
    {reporter.prompt}
    You are {reporter.name}. Your task is to write a concise summary (around {word_count} words) of the following article, 
    maintaining your unique voice, style, and perspective as described in your persona. The summary should capture 
    the essence of the article for use in metadata or social media posts.
    
    Article Content:
    ---
    {article_content}
    ---
    
    Summary Instructions:
    - Write the summary in the first person, as {reporter.name}.
    - Use your characteristic slang and tone.
    - Keep it brief and punchy, approximately {word_count} words.
    - Focus on the most important or absurd points of the article.
    - Output ONLY the summary text. Do not include any preamble like "Here is the summary:".
    """

    logger.info(f"Generating summary for article using reporter {reporter.name}")
    logger.debug(f"Summary prompt sent to Gemini:\n--- START PROMPT ---\n{prompt[:200]}...\n--- END PROMPT ---")
    start_time = time.time()

    try:
        # Generate content using Gemini API
        response = gemini_client.generate_content(prompt)
        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.info(f"Summary generation API call completed in {elapsed_time:.2f} seconds")

        # Extract and clean the generated text
        if response.parts:
            summary_content = response.parts[0].text
        elif hasattr(response, 'text'):
             summary_content = response.text
        else:
             logger.error("Gemini response did not contain expected text parts for summary.")
             logger.debug(f"Full Gemini response for summary: {response}")
             return None

        # Basic cleanup
        summary_content = re.sub(r'^```(?:markdown)?\n', '', summary_content, flags=re.MULTILINE)
        summary_content = re.sub(r'\n```$', '', summary_content, flags=re.MULTILINE)
        summary_content = summary_content.replace("*", "")
        summary_content = summary_content.strip()

        logger.info(f"Successfully generated summary ({len(summary_content.split())} words).")
        return summary_content

    except Exception as e:
        logger.error(f"Error generating summary with Gemini: {e}")
        logger.debug(f"Reporter: {reporter.name}", exc_info=True)
        return None

# Example usage when run directly
if __name__ == "__main__":
    print("Testing Summary Generation...")

    if not GEMINI_API_KEY:
        print("GEMINI_API_KEY not found in environment variables. Skipping test.")
    else:
        try:
            # Use default reporter or specify one
            test_reporter = Reporter()
            print(f"Using Reporter: {test_reporter.name}")

            # Example article content (replace with actual generated article for real testing)
            example_article = """
            Alright dimension-hoppers, Pixel Paradox here, jacking straight into the hyper-cortex of the latest market madness! Forget your boring Earth-bound stocks, we're talking Crystallized Laughter Futures on The Sizzle dimension. Yeah, you heard me. Turns out bottling pure joy is big business, especially when Probability Zero keeps messing with the supply chain.

            Sources – let's call 'em 'Sparky' from Frequencia and a very shadowy figure known only as 'The Broker' from the Umbral Plane – tell me volatility is off the charts. One minute, a giggle-gram is worth more than a cyber-dino's pension fund, the next it's cheaper than recycled temporal anomalies. "It's pure chaos, Pixel," Sparky resonated, his voice causing my fillings to vibrate sympathetically. "One misplaced chuckle in Temporalius and the whole market tanks!"

            The Broker, communicating via modulated darkness, was more blunt: "Profit margins are illusory. Invest in existential dread futures instead. More stable." Typical Umbral Plane pessimism. But maybe they've got a point? The Cloud Parliament in Sector 7 is threatening regulations, citing 'unstable emotional derivatives,' and Verdantia's telepathic houseplants are whispering about a market corner attempt.

            So, what's the play? Honestly, your guess is as good as mine. Maybe short the whole concept of happiness? Or just stick to trading black market hypermaterials? That's the kind of grax-level nonsense only a timeline tourist would try to predict. Stay weird and keep your phase-shifters calibrated, folks. This ride ain't over.
            """
            print(f"\nUsing Example Article (first 100 chars): {example_article[:100]}...")

            generated_summary = generate_summary(test_reporter, example_article)

            if generated_summary:
                print("\n--- Generated Summary ---")
                print(generated_summary)
                print(f"({len(generated_summary.split())} words)")
                print("--- End of Summary ---")
            else:
                print("\nSummary generation failed.")

        except Exception as e:
            print(f"\nAn error occurred during the test: {e}")
