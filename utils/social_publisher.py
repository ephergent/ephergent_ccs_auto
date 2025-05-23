#!/usr/bin/env python3
import os
import logging
import random
from pathlib import Path
from dotenv import load_dotenv
import requests # For downloading images for Bluesky

# --- Platform Specific Imports ---
# Import platform libraries, handling errors if they are not installed
try:
    from atproto import Client as BlueskyClient, models as bluesky_models
except ImportError:
    BlueskyClient = None
    bluesky_models = None
    print("Warning: atproto package not found. Bluesky posting disabled. Install with: pip install atproto")

try:
    from farcaster import Warpcast
except ImportError:
    Warpcast = None
    print("Warning: farcaster package not found. Warpcast posting disabled. Install with: pip install farcaster")

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# --- Configuration & Credentials ---
POST_TO_SOCIAL = os.getenv('POST_TO_SOCIAL', 'false').lower() in ('true', '1', 'yes', 't')

# Bluesky
BLUESKY_USERNAME = os.getenv('BLUESKY_USERNAME')
BLUESKY_PASSWORD = os.getenv('BLUESKY_PASSWORD')

# Warpcast
MNEMONIC_ENV_VAR = os.getenv('MNEMONIC_ENV_VAR') # Note: Storing mnemonics directly in env is risky

# --- Helper Functions ---

def truncate_text(text: str, max_length: int = 300) -> str:
    """Truncate text to max_length, ending with ellipsis if truncated."""
    if len(text) <= max_length:
        return text
    # Try to cut at the last space within the limit
    truncated = text[:max_length - 1]
    last_space = truncated.rfind(' ')
    if last_space != -1:
        return truncated[:last_space] + '…'
    else:
        # If no space found, just cut at the limit
        return truncated + '…'

def generate_social_post_text(title: str, summary: str) -> str:
    """Generate text for social media post using title and summary."""
    # Fun intros for social posts (similar to old system)
    intros = [
        "BREAKING INTERDIMENSIONAL NEWS!", "Just dropped from the quantum stream!",
        "ALERT ALL DIMENSIONS!", "Reality-shattering update!", "Your neural wake-up call!",
        "Timeline alert!", "The multiverse is buzzing!", "Cross-dimensional scoop!",
        "Pocket dimension exclusive!", "Hot off the chronopress!", "Ephergent dispatch!"
    ]
    # Outros/calls to action
    outros = [
        "Phase-shift into the full story!", "Calibrate your reality filters and click!",
        "Dimension-hop to read more!", "Your quantum consciousness demands the full story!",
        "Don't let the telepathic houseplants get this info first!", "Worth at least 5 crystallized laughs!",
        "The cybernetic dinosaurs approve this message!", "Read before gravity reverses next Tuesday!",
        "Multiversal truth awaits!", "Stay grax and click through!", "Link in the bio-luminescent signature!"
    ]

    intro = random.choice(intros)
    outro = random.choice(outros)

    # Combine elements - prioritize title and summary
    # Ensure the combination doesn't exceed typical platform limits excessively before truncation
    # (Truncation will happen per-platform if needed)
    post_text = f"{intro}\n\n{title}\n\n{summary}\n\n{outro}"

    return post_text


# --- Platform Posting Functions ---

def post_to_bluesky(title: str, link: str, description: str, image_path: Path | None = None, tags: list[str] | None = None) -> bool:
    """Sends a post to Bluesky with title, link, description, and optionally an image."""
    if not BlueskyClient or not bluesky_models:
        logger.warning("Bluesky client not available (atproto not installed). Skipping Bluesky post.")
        return False
    if not BLUESKY_USERNAME or not BLUESKY_PASSWORD:
        logger.error("Bluesky credentials (BLUESKY_USERNAME, BLUESKY_PASSWORD) not found in .env. Skipping Bluesky post.")
        return False

    logger.info(f"Attempting to post to Bluesky: {title}")
    try:
        client = BlueskyClient()
        client.login(BLUESKY_USERNAME, BLUESKY_PASSWORD)
        logger.info("Bluesky login successful.")

        # Prepare image blob if image_path is provided
        image_blob = None
        if image_path and image_path.is_file():
            logger.info(f"Uploading image to Bluesky: {image_path}")
            try:
                with open(image_path, 'rb') as f:
                    img_data = f.read()
                response = client.upload_blob(img_data)
                image_blob = response.blob
                logger.info("Image uploaded successfully.")
            except Exception as img_e:
                logger.error(f"Failed to upload image to Bluesky: {img_e}")
                # Continue without image if upload fails

        # Prepare text content - Bluesky has a 300 grapheme limit
        # Add hashtags to the description *before* truncation
        post_description = description
        if tags:
            hashtags = " ".join([f"#{tag.replace(' ', '').replace('-', '')}" for tag in tags if tag]) # Basic tag cleanup
            post_description += f"\n\n{hashtags}"

        # Truncate the main text content (title + link might be added separately or within embed)
        # Let's put the core message in the text part and use the embed for link details
        # Max length is 300, leave space for link potentially added by client? Be conservative.
        post_text = truncate_text(post_description, max_length=280) # Truncate description for main text

        # Create embed card for the link
        embed_external = bluesky_models.AppBskyEmbedExternal.Main(
            external=bluesky_models.AppBskyEmbedExternal.External(
                uri=link,
                title=title, # Use full title in embed
                description=truncate_text(description, max_length=200), # Shorter description for embed
                thumb=image_blob # Attach uploaded image blob as thumbnail
            )
        )

        # Combine text and embed
        embed_arg = embed_external
        if image_blob and not embed_external.external.thumb:
             # If we uploaded an image but it didn't attach to the link card (e.g., link had its own thumb)
             # We might want to attach it as a separate image embed.
             # This logic can get complex depending on desired layout.
             # For simplicity, we prioritize thumb on the link card.
             logger.warning("Image uploaded but might not be attached to link card if link provided its own preview.")
             # Example of attaching image separately (would replace embed_external):
             # embed_arg = bluesky_models.AppBskyEmbedImages.Main(images=[bluesky_models.AppBskyEmbedImages.Image(alt=title, image=image_blob)])


        # Send the post
        logger.info(f"Sending post to Bluesky. Text length: {len(post_text)}")
        client.send_post(text=post_text, embed=embed_arg)

        logger.info(f"Post sent successfully to Bluesky: {title}")
        return True

    except Exception as e:
        logger.error(f"Error sending post to Bluesky: {e}", exc_info=True)
        return False


def post_to_warpcast(title: str, link: str, status: str, tags: list[str] | None = None) -> bool:
    """Posts to Warpcast (Farcaster)."""
    if not Warpcast:
        logger.warning("Warpcast client not available (farcaster not installed). Skipping Warpcast post.")
        return False
    if not MNEMONIC_ENV_VAR:
        logger.error("Warpcast mnemonic (MNEMONIC_ENV_VAR) not found in .env. Skipping Warpcast post.")
        return False

    logger.info(f"Attempting to post to Warpcast: {title}")
    try:
        # Initialize client with mnemonic
        # Consider security implications of using mnemonic directly
        client = Warpcast(mnemonic=MNEMONIC_ENV_VAR)
        logger.info("Warpcast client initialized.") # Note: Actual login/auth happens implicitly on post

        # Prepare status message
        post_status = status
        if tags:
            hashtags = " ".join([f"#{tag.replace(' ', '').replace('-', '')}" for tag in tags if tag])
            post_status = f"{post_status}\n\n{hashtags}"

        # Warpcast character limit is higher (around 320 bytes, but varies)
        # Truncation might still be needed, but less aggressive than Bluesky
        # The library might handle truncation, or fail if too long. Let's truncate defensively.
        final_status = truncate_text(post_status, max_length=300)

        # Convert parameters to expected types
        if link:
            link = [link]

        # Send the cast
        # Note: The library might evolve; check its documentation for the latest API.
        # As of farcaster 0.7.11, it seems to be client.post_cast(...)
        logger.info(f"Sending cast to Warpcast. Status length: {len(final_status)}")
        response = client.post_cast(text=final_status, embeds=link)

        # Check response - library might return Cast object on success, None or raise error on failure
        if response and hasattr(response, 'hash'):
            logger.info(f"Successfully posted to Warpcast: {title} (Cast hash: {response.hash})")
            return True
        else:
            # If response is None or doesn't look like success, log it
            logger.error(f"Failed to post to Warpcast. Response: {response}")
            return False

    except Exception as e:
        logger.error(f"Error posting to Warpcast: {e}", exc_info=True)
        return False

# --- Main Orchestration Function ---

def post_article_to_social_media(
    title: str,
    url: str, # Link to the published article (e.g., Pelican URL)
    summary: str,
    image_path: Path | None = None, # Path to local feature image
    tags: list[str] | None = None
    ) -> dict[str, bool]:
    """
    Posts article details to configured social media platforms.

    Args:
        title: The article title.
        url: The canonical URL of the published article.
        summary: A short summary of the article.
        image_path: Path to the feature image file for upload.
        tags: List of relevant tags/hashtags.

    Returns:
        A dictionary indicating success/failure for each platform.
    """
    results = {"bluesky": False, "warpcast": False}

    if not POST_TO_SOCIAL:
        logger.info("Social media posting is disabled (POST_TO_SOCIAL is not 'true'). Skipping.")
        return results # Return all false as nothing was attempted

    logger.info(f"--- Starting Social Media Publishing for: {title} ---")
    logger.info(f"Article URL: {url}")

    # Generate the base text for posts
    post_text = generate_social_post_text(title, summary)

    # Post to Bluesky
    results["bluesky"] = post_to_bluesky(title, url, post_text, image_path, tags)

    # Post to Warpcast
    results["warpcast"] = post_to_warpcast(title, url, post_text, tags)

    logger.info(f"--- Social Media Publishing Finished ---")
    logger.info(f"Results: {results}")
    return results


# Example usage when run directly
if __name__ == "__main__":
    print("Testing Social Publisher...")

    if not POST_TO_SOCIAL:
        print("POST_TO_SOCIAL is not enabled in .env. Skipping actual posting.")
    else:
        print("POST_TO_SOCIAL is enabled. Attempting to post test data...")
        # Dummy data for testing
        test_title = "Social Media Test: Sentient Clouds Demand Representation"
        # Use a real, accessible URL for testing link embeds
        test_url = "https://example.com/article/sentient-clouds" # Replace with a real URL if possible
        test_summary = "Pixel Paradox reporting: The Cloud Parliament in Sector 7 is in uproar! Sentient weather patterns demand voting rights. Will it rain legislation or hailstones of fury? #Ephergent #DimensionalPolitics"
        test_tags = ["Ephergent", "DimensionalPolitics", "Sector7", "Test"]

        # Create a dummy image file for testing upload
        test_output_dir = Path("./output")
        test_output_dir.mkdir(exist_ok=True)
        dummy_image_path = test_output_dir / "social_test_image.png"
        try:
            from PIL import Image
            img = Image.new('RGB', (60, 30), color = 'red')
            img.save(dummy_image_path)
            print(f"Created dummy image: {dummy_image_path}")
        except ImportError:
            print("Pillow not installed, cannot create dummy image for testing.")
            dummy_image_path = None
        except Exception as img_e:
            print(f"Error creating dummy image: {img_e}")
            dummy_image_path = None


        # Run the main function
        results = post_article_to_social_media(
            title=test_title,
            url=test_url,
            summary=test_summary,
            image_path=dummy_image_path,
            tags=test_tags
        )

        print("\n--- Test Results ---")
        print(f"Bluesky: {'Success' if results['bluesky'] else 'Failed/Skipped'}")
        print(f"Warpcast: {'Success' if results['warpcast'] else 'Failed/Skipped'}")

        # Clean up dummy image
        if dummy_image_path and dummy_image_path.exists():
            try:
                dummy_image_path.unlink()
                print(f"Removed dummy image: {dummy_image_path}")
            except OSError as e:
                print(f"Error removing dummy image: {e}")
