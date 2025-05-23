#!/usr/bin/env python3
import os
import argparse
import requests
import json
import logging
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

def send_email(subject: str, template: str, variables: dict, recipient: str | None = None):
    """
    Send an email using Mailgun API.

    Args:
        subject (str): Email subject line.
        template (str): Name of the Mailgun template to use.
        variables (dict): Dictionary of variables for the Mailgun template.
                          Based on newsletter_template.html, expects keys like:
                          - newsletter_date (str): e.g., "August 15, 2024"
                          - article_title (str): Title of the article.
                          - article_summary (str): Summary text of the article.
                          - article_url (str): Absolute URL to the full article.
                          - article_feature_image_url (str): Absolute URL to the article's feature image.
        recipient (str | None): Override recipient email address. Defaults to MAILGUN_TO env var.


    Returns:
        requests.Response | None: API response or None if sending failed before request.
    """
    # Load environment variables from .env file
    load_dotenv()

    # Get API key from environment variable with fallback
    api_key = os.getenv('MAILGUN_API_KEY')
    if not api_key:
        logger.error("MAILGUN_API_KEY environment variable is missing")
        return None

    # Mailgun API endpoint
    url = os.getenv('MAILGUN_API_URL')
    if not url:
        logger.error("MAILGUN_API_URL environment variable is missing")
        return None

    mailgun_from = os.getenv('MAILGUN_FROM', "Ephergent postmaster <postmaster@email.ephergent.com>")
    mailgun_to = recipient or os.getenv('MAILGUN_TO')
    if not mailgun_to:
        logger.error("Recipient email address is missing (MAILGUN_TO env var or recipient parameter)")
        return None

    # Prepare the custom variables as JSON string
    # Variables are defined by the specific Mailgun template being used.
    variables_json = json.dumps(variables)

    # Prepare payload
    payload = {
        "from": mailgun_from,
        "to": mailgun_to,
        "subject": subject,
        "template": template,
        "h:X-Mailgun-Variables": variables_json
    }

    response = None # Initialize response to None
    try:
        # Send request to Mailgun API
        logger.info(f"Sending email via Mailgun. Subject: '{subject}', Template: '{template}', To: '{mailgun_to}'")
        response = requests.post(
            url,
            auth=("api", api_key),
            data=payload,
            timeout=30 # Add a timeout
        )
        response.raise_for_status() # Raises HTTPError for bad responses (4xx or 5xx)
        logger.info(f"Email sent successfully! Status code: {response.status_code}")
    except requests.exceptions.Timeout:
        logger.error("Failed to send email: Request timed out.")
    except requests.exceptions.HTTPError as e:
        logger.error(f"Failed to send email: HTTP Error: {e.response.status_code}")
        if response is not None:
            logger.error(f"Mailgun Response: {response.text}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to send email: Request Exception: {e}")
        if response is not None:
            logger.error(f"Response Text: {response.text}") # Log response text if available
    except Exception as e:
        logger.error(f"An unexpected error occurred during email sending: {e}", exc_info=True)


    return response

def main():
    parser = argparse.ArgumentParser(description='Send a test email using a Mailgun template')

    parser.add_argument('--subject', type=str, default="Test Dispatch",
                        help='Email subject line (default: "Test Dispatch")')

    parser.add_argument('--title', type=str, default="Test Article Title",
                        help='Title of the article (default: "Test Article Title")')

    parser.add_argument('--summary', type=str,
                        default="This is the summary text for the test article. It should be a few sentences long.",
                        help='Summary text of the article')

    parser.add_argument('--url', type=str,
                        default="https://ephergent.com",
                        help='URL of the article (default: https://ephergent.com)')

    parser.add_argument('--image_url', type=str,
                        default="https://ephergent.com/theme/images/profile.png",
                        help='URL of the article featured image (default: https://ephergent.com/theme/images/profile.png)')

    parser.add_argument('--date', type=str, default="August 15, 2024",
                        help='Newsletter date string (default: August 15, 2024)')

    parser.add_argument('--recipient', type=str, default=None,
                        help='Optional recipient email address to override MAILGUN_TO')

    args = parser.parse_args()

    # Example variables matching the newsletter_template.html
    variables = {
        "newsletter_date": args.date,
        "article_title": args.title,
        "article_summary": args.summary,
        "article_url": args.url,
        "article_feature_image_url": args.image_url,
    }

    # Get template name from env or use default
    template_name = os.getenv('MAILGUN_TEMPLATE', "daily.dimensional.dispatch") # Make sure this matches your Mailgun template name

    send_email(
        subject=args.subject,
        template=template_name,
        variables=variables,
        recipient=args.recipient
    )

if __name__ == "__main__":
    # Basic logging setup for standalone execution
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    main()
