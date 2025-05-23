#!/usr/bin/env python3
"""
YouTube Video Uploader Utility

Handles authentication and uploading videos/thumbnails to YouTube using the Data API.
Adapted from ref_code/youtube_uploader.py for integration into the main app.
"""

import os
import logging
from pathlib import Path
import mimetypes
from typing import Optional, List

# Third-party libraries for Google API
try:
    import google_auth_httplib2 # Required for authorized http instance
    import google.oauth2.credentials
    import googleapiclient.discovery
    import googleapiclient.errors
    import googleapiclient.http
    from google_auth_oauthlib.flow import InstalledAppFlow
    import google.auth.transport.requests
    GOOGLE_API_AVAILABLE = True
except ImportError:
    GOOGLE_API_AVAILABLE = False
    # Define dummy classes/functions if google libs are not installed
    class DummyService:
        def videos(self): return self
        def thumbnails(self): return self
        def insert(self, *args, **kwargs): return self
        def set(self, *args, **kwargs): return self
        def execute(self, *args, **kwargs): return {'id': 'dummy_video_id'} if 'media_body' in kwargs else {}

    googleapiclient = None
    InstalledAppFlow = None
    google = None


# --- Constants ---
logger = logging.getLogger(__name__)

YOUTUBE_API_SERVICE_NAME = 'youtube'
YOUTUBE_API_VERSION = 'v3'
SCOPES = ['https://www.googleapis.com/auth/youtube.upload']

# Configuration from Environment Variables or Defaults
SECRETS_DIR = Path(os.getenv('YOUTUBE_SECRETS_DIR', Path(__file__).parent.parent / 'secrets'))
CLIENT_SECRETS_FILE = SECRETS_DIR / os.getenv('YOUTUBE_CLIENT_SECRET_FILE', 'client_secret.json')
TOKEN_FILE = SECRETS_DIR / os.getenv('YOUTUBE_TOKEN_FILE', 'token.json')
DEFAULT_CATEGORY_ID = os.getenv('YOUTUBE_CATEGORY_ID', '24') # Default to '24' entertainment
DEFAULT_PRIVACY_STATUS = os.getenv('YOUTUBE_PRIVACY_STATUS', 'unlisted') # Default to 'unlisted'

# --- Authentication ---

def get_authenticated_service():
    """Authenticates using OAuth 2.0 and returns the YouTube Data API service object."""
    if not GOOGLE_API_AVAILABLE:
        logger.error("Google API client libraries not installed. Cannot authenticate.")
        return None

    creds = None
    if TOKEN_FILE.exists():
        try:
            creds = google.oauth2.credentials.Credentials.from_authorized_user_file(str(TOKEN_FILE), SCOPES)
            logger.info(f"Loaded credentials from {TOKEN_FILE}")
        except Exception as e:
            logger.warning(f"Failed to load token file {TOKEN_FILE}: {e}. Will attempt re-authentication.")
            creds = None # Force re-authentication

    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            logger.info("Credentials expired. Refreshing token...")
            try:
                creds.refresh(google.auth.transport.requests.Request())
                logger.info("Token refreshed successfully.")
            except Exception as e:
                logger.error(f"Failed to refresh token: {e}. Need new authentication.", exc_info=True)
                creds = None # Force re-authentication
        else:
            logger.info("No valid credentials found. Starting OAuth flow...")
            if not CLIENT_SECRETS_FILE.exists():
                logger.error(f"Client secrets file not found at {CLIENT_SECRETS_FILE}. Cannot authenticate.")
                return None
            try:
                flow = InstalledAppFlow.from_client_secrets_file(str(CLIENT_SECRETS_FILE), SCOPES)
                # Note: run_local_server requires a browser environment.
                # For server environments, consider device flow or service accounts.
                creds = flow.run_local_server(port=0)
                logger.info("Authentication successful.")
            except Exception as e:
                logger.error(f"Failed to run OAuth flow: {e}", exc_info=True)
                return None

        # Save the credentials for the next run
        if creds:
            try:
                TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True) # Ensure directory exists
                with open(TOKEN_FILE, 'w') as token:
                    token.write(creds.to_json())
                logger.info(f"Credentials saved to {TOKEN_FILE}")
            except IOError as e:
                logger.error(f"Failed to save token file to {TOKEN_FILE}: {e}")

    if not creds:
        logger.error("Could not obtain valid credentials.")
        return None

    # Build the service object
    try:
        service = googleapiclient.discovery.build(
            YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, credentials=creds
        )
        logger.info("YouTube API service built successfully.")
        return service
    except Exception as e:
        logger.error(f"Failed to build YouTube API service: {e}", exc_info=True)
        return None

# --- Upload Logic ---

def upload_to_youtube(
    video_file_path: Path,
    title: str,
    description: str,
    tags: List[str],
    thumbnail_path: Optional[Path] = None,
    category_id: str = DEFAULT_CATEGORY_ID,
    privacy_status: str = DEFAULT_PRIVACY_STATUS,
    ) -> Optional[str]:
    """
    Uploads a video and optionally a thumbnail to YouTube.

    Args:
        video_file_path (Path): Path to the video file to upload.
        title (str): The title for the YouTube video.
        description (str): The description for the YouTube video.
        tags (List[str]): A list of tags for the YouTube video.
        thumbnail_path (Optional[Path]): Path to the custom thumbnail image.
        category_id (str): YouTube category ID (default: '28' Science & Technology).
        privacy_status (str): Video privacy ('private', 'unlisted', 'public').

    Returns:
        Optional[str]: The YouTube video ID if upload is successful, otherwise None.
    """
    if not GOOGLE_API_AVAILABLE:
        logger.error("Google API client libraries not installed. Cannot upload video.")
        return None

    if not video_file_path.exists():
        logger.error(f"Video file not found: {video_file_path}")
        return None
    if thumbnail_path and not thumbnail_path.exists():
        logger.warning(f"Thumbnail file not found: {thumbnail_path}. Uploading without custom thumbnail.")
        thumbnail_path = None

    youtube = get_authenticated_service()
    if not youtube:
        logger.error("Failed to get authenticated YouTube service. Cannot upload.")
        return None

    logger.info(f"Starting YouTube upload for: {video_file_path.name}")
    logger.info(f"Title: {title}")
    logger.info(f"Category ID: {category_id}, Privacy: {privacy_status}")
    logger.debug(f"Description: {description[:100]}...") # Log truncated description
    logger.debug(f"Tags: {tags}")

    body = {
        'snippet': {
            'title': title,
            'description': description,
            'tags': tags,
            'categoryId': category_id
        },
        'status': {
            'privacyStatus': privacy_status
        }
    }

    # --- Upload Video ---
    video_id = None
    try:
        logger.info("Uploading video file...")
        # Determine video MIME type
        video_mime_type, _ = mimetypes.guess_type(str(video_file_path))
        if not video_mime_type or not video_mime_type.startswith('video/'):
            video_mime_type = 'application/octet-stream' # Fallback
            logger.warning(f"Could not determine video MIME type for {video_file_path.name}. Using fallback '{video_mime_type}'.")

        media_video = googleapiclient.http.MediaFileUpload(
            str(video_file_path),
            mimetype=video_mime_type,
            resumable=True # Use resumable uploads for large files
        )
        insert_request = youtube.videos().insert(
            part='snippet,status',
            body=body,
            media_body=media_video
        )

        # Execute the request with progress reporting (optional but recommended)
        response = None
        while response is None:
            status, response = insert_request.next_chunk()
            if status:
                logger.info(f"Uploaded {int(status.progress() * 100)}%")
        video_id = response.get('id')
        if video_id:
            logger.info(f"Video uploaded successfully! Video ID: {video_id}")
        else:
            logger.error(f"Video upload failed. No ID received. Response: {response}")
            return None

    except googleapiclient.errors.HttpError as e:
        logger.error(f"An HTTP error {e.resp.status} occurred during video upload: {e.content}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred during video upload: {e}", exc_info=True)
        return None

    # --- Upload Thumbnail (if provided and video upload succeeded) ---
    if video_id and thumbnail_path:
        logger.info(f"Setting custom thumbnail from: {thumbnail_path.name}")
        try:
            # Determine thumbnail MIME type
            thumb_mime_type, _ = mimetypes.guess_type(str(thumbnail_path))
            if not thumb_mime_type or not thumb_mime_type.startswith('image/'):
                logger.error(f"Could not determine or unsupported MIME type for thumbnail: {thumbnail_path}. Skipping thumbnail upload.")
            else:
                media_thumbnail = googleapiclient.http.MediaFileUpload(
                    str(thumbnail_path),
                    mimetype=thumb_mime_type
                )
                thumbnail_request = youtube.thumbnails().set(
                    videoId=video_id,
                    media_body=media_thumbnail
                )
                thumbnail_response = thumbnail_request.execute()
                logger.info(f"Custom thumbnail set successfully! Response: {thumbnail_response}")

        except googleapiclient.errors.HttpError as e:
            logger.error(f"An HTTP error {e.resp.status} occurred during thumbnail upload: {e.content}", exc_info=True)
            # Continue, returning the video_id, as the video itself was uploaded
        except Exception as e:
            logger.error(f"An unexpected error occurred during thumbnail upload: {e}", exc_info=True)
            # Continue, returning the video_id

    return video_id

# --- Main guard for potential standalone testing ---
if __name__ == "__main__":
    print("Running youtube_uploader.py standalone for testing.")
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    if not GOOGLE_API_AVAILABLE:
        print("Error: Google API client libraries not found. Cannot run test.")
    else:
        # --- Dummy Data for Testing ---
        # IMPORTANT: Replace with actual paths to a test video and thumbnail
        test_video_file = Path("./test_video_output/test_video_gen_video.mp4") # Example path from video_generator test
        test_thumbnail_file = Path("./test_video_output/dummy_feat.png") # Example path from video_generator test

        if not test_video_file.exists():
            print(f"ERROR: Test video file not found at {test_video_file}")
            print("Please create a test video (e.g., by running utils/video_generator.py standalone)")
            print("or update the 'test_video_file' path in this script.")
        elif not test_thumbnail_file.exists():
             print(f"ERROR: Test thumbnail file not found at {test_thumbnail_file}")
             print("Please create a test image or update the 'test_thumbnail_file' path.")
        else:
            test_title = "Ephergent Test Upload"
            test_description = "This is a test video uploaded by the Ephergent Content Creation System."
            test_tags = ["ephergent", "test", "api", "python"]
            # Uses defaults for category_id and privacy_status from constants

            # --- Run Uploader ---
            print("\nCalling upload_to_youtube...")
            uploaded_id = upload_to_youtube(
                video_file_path=test_video_file,
                title=test_title,
                description=test_description,
                tags=test_tags,
                thumbnail_path=test_thumbnail_file
                # category_id='22', # Example override
                # privacy_status='private' # Example override
            )

            if uploaded_id:
                print(f"\n--- Test SUCCESS ---")
                print(f"Video uploaded with ID: {uploaded_id}")
                print(f"Link: https://www.youtube.com/watch?v={uploaded_id}")
            else:
                print(f"\n--- Test FAILED ---")
                print("Video upload did not complete successfully. Check logs above.")
                print(f"Ensure '{CLIENT_SECRETS_FILE}' exists and is valid.")
                print(f"Ensure '{TOKEN_FILE}' is valid or delete it to re-authenticate.")
