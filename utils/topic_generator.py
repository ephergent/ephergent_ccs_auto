# utils/topic_generator.py
# !/usr/bin/env python3
import os
import logging
import re
import time
import random
from pathlib import Path
from dotenv import load_dotenv

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
GEMINI_MODEL = os.getenv('GEMINI_MODEL', 'gemini-2.0-flash')  # Updated to a common model, adjust if needed
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
gemini_client = None

if genai and GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        # Configuration for generation - you can adjust temperature for more creativity
        generation_config = genai.types.GenerationConfig(
            temperature=0.8,  # Higher temperature (0.7-1.0) for more creativity
            # top_p=0.95, # Can also adjust top_p
            # top_k=40,   # Can also adjust top_k
        )
        gemini_client = genai.GenerativeModel(
            GEMINI_MODEL,
            generation_config=generation_config
        )
        logger.info(f"Gemini client initialized successfully with {GEMINI_MODEL}.")
    except Exception as e:
        logger.error(f"Failed to initialize Gemini client: {e}")
elif not genai:
    logger.error("Gemini client cannot be initialized because the 'google-generativeai' package is missing.")
elif not GEMINI_API_KEY:
    logger.error("Gemini client cannot be initialized because GEMINI_API_KEY is not set in .env.")

# --- Ephergent Universe Elements for Prompt Injection ---
# These can be expanded significantly from your ephergent_universe_prompt.md
CORE_DIMENSIONS = ["Prime Material", "Nocturne Aeturnus", "Cogsworth Cogitarium", "Verdantia", "The Edge"]
UNIVERSAL_CONSTANTS_AND_QUIRKS = [
    "Third-Tuesday Gravity Reversals",
    "Cybernetically enhanced dinosaurs running the banking system",
    "Crystallized Laughter (CLX) as currency",
    "The shadow government of telepathic houseplants",
    "A1, the quantum-computing robot assistant espresso machine",
    "Nocturne's emotion refineries",
    "Cogsworth's clocks running backward on weekends",
    "Verdantia's photosynthetic consensus",
    "The Edge's reality anchoring",
    "Prime Material's Department of Reality Maintenance (DRM)",
    "Sentient garden gnomes",
    "Floating architecture in Prime Material",
    "The visual manifestation of reality glitches (pixelated distortions)",
    "Obsidian-like stone architecture in Nocturne",
    "Continent-spanning gear-systems in Cogsworth",
    "Bio-luminescent pulses in Verdantia's root networks",
    "Shimmering, unstable spheres of light at The Edge",
    "The smell of ozone and burnt toast near paradoxes"
]


def generate_random_topic() -> str:
    """
    Selects and returns a random topic from the predefined list,
    aligned with the fun, absurd Ephergent universe.

    Returns:
        str: A randomly chosen topic.
    """
    fallback_topics = [
        "The Great Third-Tuesday Gravity Reversal Bake-Off: Floating Cakes and Upside-Down Judges",
        "Nocturne Aeturnus Fashion Week: Are Capes That Billow in Still Air the New Black?",
        "Cogsworth Cogitarium's Annual Reverse Clockwork Race: Who Can Finish Last First?",
        "Prime Material's Department of Reality Maintenance Investigates a Spontaneous Disco Ball Anomaly",
        "The Edge's Latest Reality Art Installation: Is That a Sentient Puddle or Just a Puddle?",
        "Cybernetic Dinosaur Bankers Introduce New 'Claw-Friendly' ATM Interfaces",
        "CLX Futures Market Sees Spike After Dimension-Wide Tickle-Fight Flash Mob",
        "Investigating Reports of Sentient Garden Gnomes Forming a Neighborhood Watch in Prime Material",
        "The Mystery of the Missing Cogsworthian Teacups: A Case of Temporal Theft or Just Misplaced?",
        "Verdantian Elders Issue Telepathic Warning About Overly Enthusiastic Lawn Gnomes",
        "Are Cybernetic Dinosaurs Secretly Funding the Telepathic Houseplant Shadow Government?",
        "New Study Shows Third-Tuesday Gravity Reversals Improve Office Productivity (If You Can Find Your Desk)",
        "Nocturne's Emotion Refineries Report a Surplus of 'Mild Annoyance' Essence",
        "Cogsworthian Artisans Develop a Clockwork Device That Brews Tea Backwards",
        "Verdantia's Root Network Experiences a 'Telepathic Traffic Jam' Due to Excessive Gossip",
        "Edgewalkers Debate the Ethics of Anchoring a Reality Bubble Shaped Like a Giant Donut",
        "A1 Develops Espresso Foam Art That Predicts Stock Market Crashes (Mostly)",
        "Prime Material Poodle Population Experiences Unexplained Increase in Gravity-Defying Floof",
        "Nocturne Aeturnus Goth Convention Features 'Most Melancholy Mime' Competition",
        "Cogsworth Cogitarium's Grand Pendulum Clock Starts Telling Time Sideways",
        "The Edge Reports a New Dimension Forming That Appears to Be Made Entirely of Left Socks",
        "CLX Exchange Rate Fluctuates Wildly Following Reports of a Rogue Tickle-Fight Syndicate",
        "Cybernetic T-Rex Banker Caught Hoarding CLX in a Giant, Chrome Nest",
        "Telepathic Houseplant Network Issues Urgent Bulletin About Overwatering in Sector 7G",
    ]
    selected_topic = random.choice(fallback_topics)
    logger.info(f"Generated random fallback topic: {selected_topic}")
    return selected_topic


def generate_topic(reporter: Reporter):
    """
    Generates a creative and unique news topic for The Ephergent,
    emphasizing novelty and drawing from specific universe elements.

    Args:
        reporter: An initialized Reporter object.

    Returns:
        A string containing the generated news topic, or None if generation failed.
    """
    if not reporter or not reporter.prompt or not reporter.topics:
        logger.error("Invalid reporter, missing reporter prompt, or missing topics. Cannot generate topic.")
        return None

    reporter_topics_str = ", ".join(reporter.topics)
    system_prompt = load_system_prompt()  # Assuming this loads your detailed universe guide

    # --- Inject dynamic elements for creativity ---
    # 1. Choose a primary dimension
    primary_dimension = random.choice(CORE_DIMENSIONS)

    # 2. Choose one or two specific Ephergent elements to feature
    num_elements_to_inject = random.randint(1, 2)
    injected_elements = random.sample(UNIVERSAL_CONSTANTS_AND_QUIRKS, num_elements_to_inject)
    injected_elements_str = " and ".join(injected_elements)
    if not injected_elements:  # Fallback if sampling fails (should not happen with current list sizes)
        injected_elements_str = random.choice(UNIVERSAL_CONSTANTS_AND_QUIRKS)

    # 3. Vary the "ask" slightly
    creative_angle_prompts = [
        "Propose a bizarre news story or investigation that connects these elements in an unexpected way.",
        "What if these Ephergent elements collided or interacted in a strange new manner? Generate a headline for this event.",
        "Uncover a mystery or an absurd claim involving these elements that demands journalistic investigation.",
        "Craft a topic that showcases a humorous or paradoxical situation arising from these specific universe details.",
        "Generate a news hook that would make readers question reality, based on these elements."
    ]
    creative_angle = random.choice(creative_angle_prompts)

    prompt = f"""
    {system_prompt}

    MISSION:
    You are an exceptionally creative topic generator for The Ephergent, the multiverse's premier news publication.
    Your specialty is generating highly original, fun, absurd, and quirky news topics suitable for a young adult audience (age 13-14).
    These topics must be deeply rooted in the Ephergent universe's unique lore, dimensions, and phenomena.
    The goal is to AVOID generic or repetitive ideas and instead produce topics that are surprising, visually rich, and spark curiosity.

    TASK:
    Generate a SINGLE, concise, and compelling news topic for a reporter whose expertise covers: {reporter_topics_str}.

    The topic MUST:
    1.  Be primarily set in or significantly involve the dimension: **{primary_dimension}**.
    2.  Prominently feature or be inspired by: **{injected_elements_str}**.
    3.  Align with the reporter's areas of expertise: **{reporter_topics_str}**.
    4.  Be inherently fun, strange, or lighthearted, even if reported with journalistic seriousness.
    5.  Suggest a strong visual element or an absurd scenario.
    6.  Sound like a credible, albeit bizarre, headline or story lead from The Ephergent.

    CREATIVITY GUIDELINES:
    -   **NOVELTY IS KEY**: Combine the given dimension, Ephergent elements, and reporter topics in a way that hasn't been seen before. Think of unexpected interactions or consequences.
    -   **VISUALIZE THE ABSURD**: What's the most striking or humorous image this combination could create?
    -   **AVOID REPETITION**: Do not simply rehash previous topic structures. Strive for unique angles.
    -   **EMBRACE THE QUIRKY**: The Ephergent universe is weird. Let the topics reflect that!

    Consider this creative angle: {creative_angle}

    EXAMPLES OF HIGHLY CREATIVE TOPICS (for inspiration on combining elements, not direct copying):
    -   "Prime Material DRM Deploys Sentient Garden Gnomes to Combat Rogue Reality Glitches After A1 Predicts Existential Sock Puppetry." (Focus: Prime Material, Sentient Garden Gnomes, A1, Reality Glitches)
    -   "Cogsworth Cogitarium's Backward Weekend Clocks Accidentally Unwind Cybernetic Dinosaur's Latest CLX Investment Scheme." (Focus: Cogsworth, Backward Clocks, Cybernetic Dinosaurs, CLX)
    -   "Verdantian Elders Complain Telepathic Houseplants Are Using Root Network for Unauthorized Interdimensional Karaoke Nights, Sparking Debate on Nocturne's Emotion Refineries." (Focus: Verdantia, Telepathic Houseplants, Nocturne, Emotion Refineries)
    -   "The Edge's Reality Anchoring Techniques Failing Against a Wave of Spontaneous Sentient Puddles Demanding Representation in the Cybernetic Dinosaur Banking System." (Focus: The Edge, Reality Anchoring, Sentient Puddles, Cybernetic Dinosaurs)

    Your output should be ONLY the single, concise news topic string. Do not include any preamble, explanation, or quotation marks around the topic itself unless they are part of the topic.
    """

    logger.info(
        f"Generating topic for reporter {reporter.name} (Topics: {reporter_topics_str}) with injected elements: Dimension='{primary_dimension}', Elements='{injected_elements_str}'")
    logger.debug(
        f"Topic prompt sent to Gemini:\n--- START PROMPT ---\n{prompt[:500]}...\n--- END PROMPT ---")  # Log more of the prompt
    start_time = time.time()

    if gemini_client:
        try:
            response = gemini_client.generate_content(prompt)
            end_time = time.time()
            elapsed_time = end_time - start_time
            logger.info(f"Topic generation API call completed in {elapsed_time:.2f} seconds")

            if response.parts:
                topic_text = "".join(part.text for part in response.parts)  # Concatenate if multiple parts
            elif hasattr(response, 'text') and response.text:
                topic_text = response.text
            else:  # Handle cases where response.text might be None or empty
                # Check candidate and content parts if standard text extraction fails
                if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                    topic_text = "".join(part.text for part in response.candidates[0].content.parts)
                else:
                    logger.error("Gemini response did not contain expected text parts for topic.")
                    logger.debug(f"Full Gemini response for topic: {response}")
                    return generate_random_topic()  # Fallback to random topic

            cleaned_topic = topic_text.strip()
            # Further clean-up: remove potential "Topic:" prefixes or similar boilerplate if LLM adds it
            cleaned_topic = re.sub(r"^(Topic:|Headline:|Story Lead:)\s*", "", cleaned_topic, flags=re.IGNORECASE)

            # Remove surrounding quotes if any
            if cleaned_topic.startswith('"') and cleaned_topic.endswith('"'):
                cleaned_topic = cleaned_topic[1:-1]
            elif cleaned_topic.startswith("'") and cleaned_topic.endswith("'"):
                cleaned_topic = cleaned_topic[1:-1]

            logger.info(f"Successfully generated topic: '{cleaned_topic}'")
            return cleaned_topic

        except Exception as e:
            logger.error(f"Error generating topic with Gemini: {e}")
            logger.debug(f"Reporter: {reporter.name}, Topics: {reporter_topics_str}", exc_info=True)
            return generate_random_topic()  # Fallback to random topic
    else:
        logger.error("Gemini client not initialized. Cannot generate topic. Falling back to random.")
        return generate_random_topic()  # Fallback to random topic


if __name__ == '__main__':
    print("Testing Enhanced Topic Generation...")


    # --- Mock Reporter and System Prompt for standalone testing ---
    # In your actual application, Reporter and load_system_prompt would come from your existing utils
    class MockReporter:
        def __init__(self, identifier, name, topics, prompt_text="Mock prompt"):
            self.id = identifier
            self.name = name
            self.topics = topics
            self.prompt = prompt_text  # Reporter's own persona prompt


    def mock_load_system_prompt():
        # Return a truncated version of your ephergent_universe_prompt.md for testing
        # In a real scenario, this would load your full, detailed universe guide.
        return """
            # EPHERGENT UNIVERSE SYSTEM PROMPT (ABBREVIATED FOR TEST)
            You are a storytelling assistant for The Ephergent universe.
            ## DIMENSIONAL FRAMEWORK (5 CORE DIMENSIONS)
            1. Prime Material: Base reality, unpredictable physics. Quirk: Gravity reverses every third Tuesday.
            2. Nocturne Aeturnus: Gothic, perpetual twilight. Quirk: Inhabitants faint at blood.
            3. Cogsworth Cogitarium: Steampunk, complex clockwork. Quirk: Clocks run backward on weekends.
            4. Verdantia: Ecological Fantasy, telepathic plants. Quirk: Houseplant shadow government.
            5. The Edge: Reality-Bending, new dimensions born. Quirk: Culture of impermanence.
            ## UNIVERSAL CONSTANTS
            - Physics is a suggestion.
            - Cybernetic dinosaurs run banking.
            - CLX is currency.
            - A1 is the AI espresso machine.
            (And many more details as per the full ephergent_universe_prompt.md)
        """


    # Replace actual utility function loads with mocks for this test
    original_load_system_prompt = load_system_prompt
    # Assuming Reporter class is defined elsewhere and imported
    # For this test, we'll use MockReporter

    # Patch the load_system_prompt for testing if it's complex
    # For simplicity, we'll assume load_system_prompt() can be called directly
    # If Reporter class relies on file loading not available here, MockReporter is better.

    if not GEMINI_API_KEY:
        print("GEMINI_API_KEY not found in environment variables. Skipping API call test, using random fallback.")
        # Test the fallback directly
        print("\n--- Testing Random Fallback Topic (No API Key) ---")
        topic_random = generate_random_topic()
        print(f"Generated random fallback topic: {topic_random}")
    else:
        # Replace the actual load_system_prompt with the mock for testing
        # This is a simple way to do it; for more complex scenarios, consider `unittest.mock.patch`
        globals()['load_system_prompt'] = mock_load_system_prompt

        try:
            reporters_data = [
                ("pixel_paradox", "Pixel Paradox", ["general", "news", "opinion", "feature", "investigation"]),
                ("vex_parallax", "Vex Parallax", ["science", "research", "discovery", "technology", "physics"]),
                ("nova_blacklight", "Nova Blacklight", ["entertainment", "celebrities", "culture", "arts", "music"]),
                ("zephyr_glitch", "Zephyr Glitch", ["tech", "digital", "online", "virtual", "cybersecurity", "ai"]),
                ("echo_voidwhisper", "Echo Voidwhisper", ["business", "market", "finance", "economy", "trade"]),
            ]

            for i in range(5):  # Generate a few topics to see variety
                print(f"\n--- Test Iteration {i + 1} ---")
                reporter_id, reporter_name, reporter_topics_list = random.choice(reporters_data)
                test_reporter = MockReporter(identifier=reporter_id, name=reporter_name, topics=reporter_topics_list)

                print(
                    f"Attempting to generate topic for: {test_reporter.name} (Topics: {', '.join(test_reporter.topics)})")
                topic = generate_topic(test_reporter)
                print(f"Generated topic: {topic}")
                if topic is None or topic in generate_random_topic.__defaults__[0] if hasattr(generate_random_topic,
                                                                                              '__defaults__') and generate_random_topic.__defaults__ else False:  # Crude check if it's a fallback
                    print("Warning: Topic might be a fallback or generation failed.")

            print("\n--- Testing Random Fallback Topic (API Key Present) ---")
            topic_random = generate_random_topic()
            print(f"Generated random fallback topic: {topic_random}")

        except Exception as e:
            print(f"\nAn error occurred during the test: {e}")
        finally:
            # Restore original function if patched
            globals()['load_system_prompt'] = original_load_system_prompt

