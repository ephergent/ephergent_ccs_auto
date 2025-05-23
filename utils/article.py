#!/usr/bin/env python3
import os
import logging
import re
import time
import random
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime  # Import datetime

# Try to import genai, handle import error if needed
try:
    import google.generativeai as genai
except ImportError:
    print("Error: google-generativeai package not found. Please install it: pip install google-generativeai")
    genai = None  # Set to None if import fails

# Local imports
from utils.reporter import Reporter  # Assuming this class exists and works
from utils.system_prompt import load_system_prompt  # Assuming this function exists

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# --- Gemini Client Initialization ---
GEMINI_MODEL = os.getenv('GEMINI_MODEL', 'gemini-2.0-flash')  # Using a common, capable model
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
gemini_client = None

if genai and GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        # Configuration for generation - adjust temperature for creativity/coherence balance
        generation_config = genai.types.GenerationConfig(
            temperature=0.75,  # 0.7-0.85 often good for creative writing
            # top_p=0.95,
            # top_k=40,
        )
        gemini_client = genai.GenerativeModel(
            GEMINI_MODEL,
            generation_config=generation_config
            # safety_settings adjusted if needed, but start with defaults
        )
        logger.info(f"Gemini client initialized successfully with {GEMINI_MODEL}.")
    except Exception as e:
        logger.error(f"Failed to initialize Gemini client: {e}")
elif not genai:
    logger.error("Gemini client cannot be initialized because the 'google-generativeai' package is missing.")
elif not GEMINI_API_KEY:
    logger.error("Gemini client cannot be initialized because GEMINI_API_KEY is not set in .env.")


def generate_article(
        reporter: Reporter,
        topic: str,
        word_count: int = 1500,  # Increased default for more depth
        similar_articles_context: list | None = None
) -> dict | None:
    """
    Generates an article based on the reporter's persona and a given topic using the Gemini API,
    emphasizing visual storytelling and deep universe integration.
    Returns a dictionary including the article content and metadata.

    Args:
        reporter (Reporter): An initialized Reporter object.
        topic (str): The topic for the article.
        word_count (int): Approximate desired word count.
        similar_articles_context (list | None): Optional list of dicts representing similar articles.

    Returns:
        dict | None: A dictionary containing 'content' (str), 'featured_characters' (list[str]),
                     and 'stardate' (str), or None if generation failed.
    """
    if not gemini_client:
        logger.error("Gemini client is not available. Cannot generate article.")
        return None
    if not reporter or not reporter.prompt:  # reporter.prompt is the persona prompt
        logger.error("Invalid reporter or missing reporter persona prompt. Cannot generate article.")
        return None

    # Load the main Ephergent Universe System Prompt (which includes visual storytelling guidelines)
    # This should be the full content of your 'ephergent_universe_prompt.md'
    ephergent_system_prompt = load_system_prompt()

    def _format_similarity_context(similar_articles: list | None) -> str:
        if not similar_articles:
            return "No specific prior articles on this exact angle have been flagged. Blaze a new trail!"
        context = "\n\nCONTEXT FROM PREVIOUSLY PUBLISHED RELATED ARTICLES (FOR YOUR AWARENESS):\n"
        context += "Your new article should build upon, contrast with, or offer a fresh perspective on these. AVOID mere repetition. Bring something NEW.\n"
        for i, article_data in enumerate(similar_articles[:2]):  # Limit to 2 for brevity
            meta = article_data.get('metadata', {})
            title = meta.get('title', article_data.get('id', 'Unknown Title'))
            # Consider adding a brief summary or key takeaway if available in metadata
            summary_snippet = meta.get('summary', 'No summary available.')
            if len(summary_snippet) > 150:  # Keep it brief
                summary_snippet = summary_snippet[:147] + "..."
            context += f"- Prior Article Title: \"{title}\"\n  (Note: This article explored: {summary_snippet})\n"
        return context

    similarity_context_str = _format_similarity_context(similar_articles_context)

    # Construct the detailed prompt for Gemini
    # The reporter.prompt (persona) is integrated directly into the main mission.
    prompt = f"""
{ephergent_system_prompt}

## YOUR MISSION AS AN EPHERGENT REPORTER ##

You are **{reporter.name}**, a seasoned journalist for *The Ephergent*. Your specific voice, quirks, dimensional origins, and reporting style are defined by your personal dossier:
<ReporterPersonaDossier>
{reporter.prompt}
</ReporterPersonaDossier>

Your assignment is to write a **brand new, original article** of approximately **{word_count} words** for our young adult audience (ages 13-14).

## Today's Topic: {topic} ##

{similarity_context_str}

## CRITICAL ARTICLE REQUIREMENTS ##

1.  **EMBODY YOUR PERSONA (First-Person Narrative):**
    * Write entirely in the first person, fully inhabiting the character of {reporter.name}.
    * Your unique voice, mannerisms, biases, and any signature phrases (use sparingly and naturally) must shine through.
    * Let your dimensional background and personal history color your observations and interpretations.

2.  **VISUAL STORYTELLING IS PARAMOUNT (The Cinematic Eye):**
    * **This is non-negotiable.** Adhere strictly to the "VISUAL STORYTELLING ENHANCEMENT" and "DIMENSIONAL AESTHETIC GUIDES" sections within the Ephergent Universe System Prompt provided above.
    * Paint vivid, dynamic, multi-sensory pictures. Describe scenes with cinematic detail: framing, lighting, texture, movement.
    * **Show, don't just tell.** Instead of "it was strange," describe *how* it looked, sounded, smelled, and felt strange, using precise vocabulary and striking metaphors.
    * Make the impossible tangible through hyper-descriptive sensory language.

3.  **DEEP UNIVERSE INTEGRATION:**
    * Naturally weave in specific elements from The Ephergent universe:
        * **Dimensional Slang:** Incorporate slang from at least two different dimensions (e.g., Cogsworthian clockwork curses, Verdantian growth-metaphors, Nocturne's poetic gloom, Prime Material's glitch-speak, Edgewalker probability-puns). Make it feel authentic to your character.
        * **Absurd Locations & Concepts:** Reference specific dimensional characteristics, unique physics, recurring elements (cybernetic dinosaurs, telepathic houseplants, CLX, gravity reversals, A1, etc.) in a way that serves the narrative.
        * **A1 Interaction (Recommended):** Consider including a brief, characteristic interaction with A1, our quantum-computing espresso machine assistant. How might A1 contribute (or amusingly complicate) your investigation or data gathering? Describe A1's visual cues (lights, steam, holographic displays).

4.  **ORIGINALITY & PLAUSIBLE ABSURDITY:**
    * Create completely fictional sources, quotes, "expert" opinions, statistics, or events that sound plausible *within* The Ephergent's absurd reality.
    * The article should be entertaining, quirky, and perhaps offer underlying biting commentary or satire relevant to the topic, all filtered through your unique perspective.

5.  **TONE & STYLE:**
    * Maintain a tone appropriate for the topic (e.g., investigative, feature, opinion) but always infused with the characteristic Ephergent blend of deadpan seriousness reporting on the utterly bizarre.
    * Your narrative should be engaging, with a clear beginning, middle, and a concluding thought or lingering question.

6.  **OUTPUT FORMAT (VERY IMPORTANT):**
    * **Output ONLY the narrative article text itself.**
    * **DO NOT include:**
        * Any metadata headers (e.g., "Filed by:", "Location:", "Stardate:", "Featured Characters:").
        * The article title.
        * Any preambles, author notes, or explanations outside the article body.
        * Placeholders or descriptions for images (e.g., "[Image: A description]"). The text must stand alone.
    * The output should begin directly with the first sentence of your article.

7.  **LENGTH & PACING:**
    * Strive for approximately {word_count} words.
    * Vary sentence structure and pacing to maintain reader engagement.

Remember, {reporter.name}, your goal is to transport the reader directly into the heart of this story with your words and unique perspective. Make them see, hear, and feel the impossible.
Now, begin your report.
"""

    logger.info(f"Generating article for topic: '{topic}' by {reporter.name} (approx. {word_count} words).")
    if similar_articles_context:
        logger.info(f"Providing similarity context: {similarity_context_str[:200]}...")  # Log snippet of context
    # logger.debug(f"Full prompt sent to Gemini:\n--- START PROMPT ---\n{prompt}\n--- END PROMPT ---") # For deep debugging, can log full prompt
    logger.debug(f"Prompt preview (first 500 chars):\n{prompt[:500]}...")

    start_time = time.time()

    try:
        response = gemini_client.generate_content(prompt)
        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.info(f"Article generation API call completed in {elapsed_time:.2f} seconds.")

        article_content = ""
        if response.parts:
            article_content = "".join(part.text for part in response.parts)
        elif hasattr(response, 'text') and response.text:
            article_content = response.text
        else:
            # More robust check for content in candidates
            if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                article_content = "".join(part.text for part in response.candidates[0].content.parts)
            else:
                logger.error("Gemini response did not contain expected text parts or candidates.")
                logger.debug(f"Full Gemini response: {response}")
                return None

        # Cleanup: Remove potential markdown code fences and leading/trailing whitespace
        article_content = re.sub(r'^```(?:markdown)?\n', '', article_content, flags=re.MULTILINE)
        article_content = re.sub(r'\n```$', '', article_content, flags=re.MULTILINE)
        article_content = article_content.strip()

        # Enhanced cleanup for any lingering metadata-like lines (already in your original code, good to keep)
        article_lines = article_content.splitlines()
        cleaned_article_body_lines = []
        unwanted_header_in_body_pattern = re.compile(
            r"^\s*(?:\*\*|__|[*])?\s*(Filed by|Location|Stardate|Featured Characters|Author|Cycle|Date|Summary|Slug|Category|Tags|Title)\s*:\s*.*",
            re.IGNORECASE
        )
        for line in article_lines:
            stripped_line = line.strip()
            if unwanted_header_in_body_pattern.match(stripped_line):
                logger.info(f"Article.py cleanup: Stripping embedded metadata-like line: '{stripped_line[:100]}...'")
                continue
            cleaned_article_body_lines.append(line)
        article_content = "\n".join(cleaned_article_body_lines).strip()

        logger.info(f"Successfully generated article content ({len(article_content.split())} words).")

        now = datetime.now()
        # This stardate format is just an example, adjust as per your universe's conventions
        stardate = f"Cycle {now.year % 100}.{now.strftime('%j')}.{random.randint(100, 999)}"

        # Basic featured character: the reporter.
        # Advanced: Could use another LLM call or regex to extract other Ephergent character names if they are mentioned.
        featured_characters = [reporter.id]
        # Example: if "A1" is in article_content, add "a1_assistant"
        if re.search(r'\bA1\b', article_content, re.IGNORECASE):
            if "a1_assistant" not in featured_characters:  # Assuming "a1_assistant" is A1's ID
                featured_characters.append("a1_assistant")

        return {
            "content": article_content,
            "featured_characters": featured_characters,
            "stardate": stardate
        }

    except Exception as e:
        logger.error(f"Error generating article with Gemini: {e}")
        # Log more details if available in the exception, e.g., response from API if it's a google.api_core.exceptions type
        if hasattr(e, 'response'):
            logger.error(f"API Response (if available): {e.response}")
        logger.debug(f"Reporter: {reporter.name}, Topic: {topic}", exc_info=True)
        return None


# Example usage when run directly
if __name__ == "__main__":
    print("Testing Enhanced Article Generation...")

    if not GEMINI_API_KEY:
        print("GEMINI_API_KEY not found in environment variables. Skipping API call test.")
    else:
        # --- Mock Reporter and System Prompt for standalone testing ---
        class MockReporter:  # Simplified for testing
            def __init__(self, identifier, name, persona_prompt_text):
                self.id = identifier
                self.name = name
                self.prompt = persona_prompt_text  # This is the key persona part


        def mock_load_system_prompt():  # Simplified for testing
            # In a real scenario, this would load your full 'ephergent_universe_prompt.md'
            return """
# EPHERGENT UNIVERSE SYSTEM PROMPT (ABBREVIATED FOR TEST)
You are a storytelling assistant for The Ephergent universe. Prioritize cinematic, visual detail.
## DIMENSIONAL FRAMEWORK
Prime Material: Base reality. Quirk: Gravity reverses every third Tuesday.
Nocturne Aeturnus: Gothic, perpetual twilight.
Cogsworth Cogitarium: Steampunk, complex clockwork.
Verdantia: Ecological Fantasy, telepathic plants.
The Edge: Reality-Bending.
## VISUAL STORYTELLING ENHANCEMENT
Think like a film director. Use dynamic staging, describe materiality and texture.
(Include more key visual guidelines from your full prompt here)
"""


        # --- End Mocks ---

        # Replace actual utility function loads with mocks for this test
        original_load_system_prompt = load_system_prompt
        globals()['load_system_prompt'] = mock_load_system_prompt
        # Assuming Reporter class is more complex, we use MockReporter for this test
        # If your Reporter class is simple and doesn't do file I/O on init, you might use it directly.

        try:
            # Example Reporter Persona (Pixel Paradox - from your personality_prompts.json snippet)
            pixel_paradox_persona = """
            You are Pixel Paradox, a sharp, tech-savvy journalist for The Ephergent.
            Your beat is general news, features, and investigations across all dimensions, but you have a soft spot for Prime Material's weirdness and anything involving digital anomalies or AI.
            You're known for your electric blue hair, punk-inspired cyber gear, and a tendency to speak in a mix of journalistic precision and glitch-hop slang.
            You're skeptical but curious, always chasing the 'data ghosts' behind the story.
            You often consult with A1, sometimes getting frustrated with its overly literal interpretations but valuing its processing power.
            Signature phrases: "Signal boosting this...", "My diagnostics are buzzing...", "That's a full system crash of logic."
            Dimensional Slang:
            - Prime Material: "That's some prime-time weird," "glitch-tastic," "reality-render error."
            - Cogsworth: "Cog-rot!" (exclamation of frustration), "chronometric anomaly."
            - The Edge: "Probability-pucked," "void-venturing."
            """
            test_reporter = MockReporter(identifier="pixel_paradox", name="Pixel Paradox",
                                         persona_prompt_text=pixel_paradox_persona)
            test_topic = "The sudden rise of sentient, graffiti-spraying drones in Prime Material and their surprisingly sophisticated art critiques."
            test_word_count = 500  # Shorter for faster testing

            # Simulate some "similar articles" context
            mock_similar_articles = [
                {"id": "drone_art_001",
                 "metadata": {"title": "Cogsworthian Automatons Debate Aesthetics of Steam-Powered Sculptures",
                              "summary": "An earlier piece on machine-generated art in a different dimension."}},
                {"id": "prime_glitch_078",
                 "metadata": {"title": "Mysterious Memetic Hazard Causes Spontaneous Interpretive Dance in Sector 7G",
                              "summary": "A report on unexplained public phenomena in Prime Material."}}
            ]

            print(f"Using Reporter: {test_reporter.name}")
            print(f"Using Topic: {test_topic}")
            print(f"Target word count: {test_word_count}")

            generated_article_data = generate_article(
                test_reporter,
                test_topic,
                word_count=test_word_count,
                similar_articles_context=mock_similar_articles
            )

            if generated_article_data and generated_article_data.get("content"):
                print("\n--- Generated Article Data (SUCCESS) ---")
                print(f"Stardate: {generated_article_data.get('stardate')}")
                print(f"Featured Characters: {generated_article_data.get('featured_characters')}")
                print("\n--- Article Content Preview (First 500 chars) ---")
                print(generated_article_data.get('content', '')[:500] + "...")
                print(
                    f"\n--- (Full article approx. {len(generated_article_data.get('content', '').split())} words) ---")

                output_dir = Path(
                    __file__).resolve().parent.parent / 'output'  # Assumes script is in a 'utils' like folder
                if not output_dir.exists():
                    output_dir.mkdir(parents=True, exist_ok=True)

                # Sanitize reporter name and topic for filename
                reporter_name_safe = re.sub(r'\W+', '_', test_reporter.name.lower())
                topic_safe = re.sub(r'\W+', '_', test_topic.lower())[:50]  # Truncate topic for filename

                test_file_path = output_dir / f"test_article_{reporter_name_safe}_{topic_safe}.md"

                with open(test_file_path, "w", encoding="utf-8") as f:
                    f.write(f"# Test Article: {test_topic}\n")
                    f.write(f"## Reporter: {test_reporter.name}\n")
                    f.write(f"### Stardate: {generated_article_data.get('stardate')}\n")
                    f.write(f"### Featured: {', '.join(generated_article_data.get('featured_characters', []))}\n\n")
                    f.write("---\n\n")
                    f.write(generated_article_data.get('content', ''))
                print(f"\nFull article data saved to: {test_file_path}")
            else:
                print("\nArticle generation failed or returned empty content.")
                if generated_article_data:
                    print(f"Returned data: {generated_article_data}")


        except Exception as e:
            print(f"\nAn error occurred during the test: {e}")
            logger.error("Error during __main__ test", exc_info=True)
        finally:
            # Restore original function if it was patched
            globals()['load_system_prompt'] = original_load_system_prompt
