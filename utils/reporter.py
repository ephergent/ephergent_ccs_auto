#!/usr/bin/env python3
import os
import json
import random
import logging
from pathlib import Path
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Default prompts file location relative to this script's directory parent
DEFAULT_PROMPTS_DIR = Path(__file__).parent.parent / 'prompts'
PROMPTS_CONFIG_FILE = DEFAULT_PROMPTS_DIR / 'personality_prompts.json'

class Reporter:
    """
    Manages reporter personalities and their attributes based on configuration files.
    """
    def __init__(self, identifier: str = None, prompts_config_path: Path = PROMPTS_CONFIG_FILE):
        """
        Initializes a reporter instance.

        Args:
            identifier (str, optional): The specific reporter ID or name to load.
                                        If None, attempts to load from REPORTER env var
                                        (expected to be an ID) or selects the default reporter.
            prompts_config_path (Path, optional): Path to the personality prompts JSON config file.
                                                  Defaults to PROMPTS_CONFIG_FILE.
        """
        self.prompts_config_path = prompts_config_path
        self.prompts_dir = prompts_config_path.parent
        self.reporters_config = self._load_reporters_config()
        self.all_reporters_data = self.reporters_config.get('reporters', [])

        selected_identifier = self._determine_reporter_identifier(identifier)
        self.reporter_data = self._find_reporter_data(selected_identifier)

        if not self.reporter_data:
            logger.error(f"Reporter with identifier '{selected_identifier}' not found and no default available. Cannot initialize.")
            # Allow initialization but log heavily. Attributes will return defaults.
            self.name = "unknown" # Default name if lookup fails
            self.id = selected_identifier or "unknown" # Store the requested ID if available
        else:
            # Use data from the found config
            self.id = self.reporter_data.get('id', 'unknown_id') # Get ID from config
            self.name = self.reporter_data.get('name', 'Unknown Reporter') # Get Name from config
            self._load_prompt_text() # Load prompt text only if reporter data is valid

        logger.info(f"Initialized reporter: ID='{self.id}', Name='{self.name}'")

    def _load_reporters_config(self) -> dict:
        """Loads the main reporters configuration JSON file."""
        try:
            with open(self.prompts_config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            if 'reporters' not in config:
                logger.error(f"'{self.prompts_config_path}' is missing the 'reporters' key.")
                return {}
            return config
        except FileNotFoundError:
            logger.error(f"Reporter config file not found: {self.prompts_config_path}")
            return {}
        except json.JSONDecodeError:
            logger.error(f"Error decoding JSON from {self.prompts_config_path}")
            return {}
        except Exception as e:
            logger.error(f"Error loading reporter config: {e}")
            return {}

    def _normalize_name(self, name: str) -> str:
        """Normalizes a reporter name for comparison (less used now, primarily for fallback)."""
        if not name: return ""
        return name.lower().strip() # Simpler normalization for name matching if needed

    def _determine_reporter_identifier(self, requested_identifier: str = None) -> str | None:
        """Determines the reporter ID to use based on request, env var, or default."""
        if requested_identifier:
            # Assume the provided identifier is the ID
            return requested_identifier

        # Check environment variable (expecting an ID)
        env_reporter_id = os.getenv('REPORTER')
        if env_reporter_id:
            logger.info(f"Using reporter ID from REPORTER environment variable: {env_reporter_id}")
            return env_reporter_id

        # Find default reporter by 'default: true' flag
        for reporter_data in self.all_reporters_data:
            if reporter_data.get('default', False):
                default_id = reporter_data.get('id')
                if default_id:
                    logger.info(f"Using default reporter ID found in config: {default_id}")
                    return default_id
                else:
                    logger.warning(f"Default reporter '{reporter_data.get('name')}' is missing an 'id'.")

        # Fallback if no default is explicitly set (e.g., use first reporter with an ID)
        if self.all_reporters_data:
            first_reporter_id = self.all_reporters_data[0].get('id')
            if first_reporter_id:
                logger.warning("No default reporter set, using the ID of the first reporter found.")
                return first_reporter_id

        logger.warning("No reporter identifier specified, no environment variable set, and no default found.")
        return None # Indicate no identifier could be determined

    def _find_reporter_data(self, identifier: str) -> dict | None:
        """Finds the configuration data primarily by ID, with fallback to normalized name."""
        if not identifier:
            return None

        # Primary lookup: by ID
        for reporter_data in self.all_reporters_data:
            if reporter_data.get('id') == identifier:
                logger.debug(f"Found reporter data by ID: {identifier}")
                return reporter_data

        # Fallback lookup: by normalized name (in case a name was passed instead of ID)
        normalized_identifier = self._normalize_name(identifier)
        logger.warning(f"Reporter ID '{identifier}' not found. Attempting fallback lookup by normalized name '{normalized_identifier}'...")
        for reporter_data in self.all_reporters_data:
             # Normalize the name from the config for comparison
             config_name = reporter_data.get('name', '')
             normalized_config_name = self._normalize_name(config_name)
             if normalized_config_name == normalized_identifier:
                 logger.warning(f"Found reporter data by matching normalized name '{normalized_identifier}' for reporter '{config_name}'. Consider using the ID '{reporter_data.get('id')}' instead.")
                 return reporter_data

        logger.warning(f"Reporter data not found for identifier (ID or Name): {identifier}")
        return None

    def _load_prompt_text(self):
        """Loads the specific prompt text for the current reporter."""
        self.prompt_text = "" # Default to empty
        if not self.reporter_data:
            # Use self.id if name isn't set yet during failed init
            reporter_id_for_log = getattr(self, 'id', 'unknown')
            logger.warning(f"Cannot load prompt text, reporter data for ID '{reporter_id_for_log}' is missing.")
            return

        prompt_filename = self.reporter_data.get('prompt_file')
        if not prompt_filename:
            logger.warning(f"No 'prompt_file' specified for reporter ID '{self.id}' (Name: '{self.name}').")
            return

        prompt_path = self.prompts_dir / prompt_filename
        try:
            with open(prompt_path, 'r', encoding='utf-8') as f:
                self.prompt_text = f.read()
            logger.debug(f"Successfully loaded prompt text for '{self.id}' from: {prompt_path}")
        except FileNotFoundError:
            logger.error(f"Prompt file not found for reporter ID '{self.id}': {prompt_path}")
        except Exception as e:
            logger.error(f"Error loading prompt text for ID '{self.id}' from {prompt_path}: {e}")

    # --- Properties to access reporter attributes ---

    @property
    def email(self) -> str:
        """Gets the email address for the current reporter."""
        return self.reporter_data.get('email', 'default_reporter@example.com') if self.reporter_data else 'default_reporter@example.com'

    @property
    def model(self) -> str:
        """Gets the preferred AI model for the current reporter."""
        # Default model can come from reporter data or model_defaults in config
        default_model = 'gemini-2.0-flash' # Hardcoded fallback default
        if self.reporters_config.get('model_defaults', {}).get('article_generator'):
             default_model = self.reporters_config['model_defaults']['article_generator']

        return self.reporter_data.get('model', default_model) if self.reporter_data else default_model

    @property
    def topics(self) -> list[str]:
        """Gets the list of topics covered by the current reporter."""
        return self.reporter_data.get('topics', ['general']) if self.reporter_data else ['general']

    @property
    def tags(self) -> list[str]:
        """Gets the default tags associated with the current reporter."""
        return self.reporter_data.get('tags', ['The Ephergent']) if self.reporter_data else ['The Ephergent']

    @property
    def stable_diffusion_prompt(self) -> str:
        """Gets the base Stable Diffusion prompt fragment for the reporter's image."""
        return self.reporter_data.get('stable_diffusion_prompt', '') if self.reporter_data else ''

    @property
    def voice(self) -> str:
        """Gets the preferred TTS voice configuration for the current reporter."""
        # Default voice can come from reporter data or model_defaults in config
        default_voice = 'alloy' # Example default, adjust as needed for your TTS
        if self.reporters_config.get('model_defaults', {}).get('tts_voice'):
             default_voice = self.reporters_config['model_defaults']['tts_voice']

        return self.reporter_data.get('voice', default_voice) if self.reporter_data else default_voice

    @property
    def prompt(self) -> str:
        """Gets the loaded personality prompt text for the current reporter."""
        return getattr(self, 'prompt_text', "") # Use getattr for safety if _load_prompt_text failed

    # --- Class Methods ---

    @classmethod
    def get_reporter_for_topic(cls, topic: str, prompts_config_path: Path = PROMPTS_CONFIG_FILE) -> 'Reporter':
        """
        Selects an appropriate reporter instance (by ID) for a given topic.

        Args:
            topic (str): The topic to find a reporter for.
            prompts_config_path (Path, optional): Path to the prompts config file.

        Returns:
            Reporter: An instance of the Reporter class suited for the topic.
        """
        # Initialize with None to load all configs, but don't select a specific reporter yet
        instance = cls(identifier=None, prompts_config_path=prompts_config_path)
        matching_reporter_ids = []
        normalized_topic = topic.lower()

        for reporter_data in instance.all_reporters_data:
            reporter_topics = [t.lower() for t in reporter_data.get('topics', [])]
            reporter_id = reporter_data.get('id')
            if normalized_topic in reporter_topics and reporter_id:
                matching_reporter_ids.append(reporter_id)

        if matching_reporter_ids:
            selected_id = random.choice(matching_reporter_ids)
            logger.info(f"Found {len(matching_reporter_ids)} reporter(s) for topic '{topic}'. Randomly selected ID: {selected_id}")
            # Return a new instance specifically for the selected reporter ID
            return cls(identifier=selected_id, prompts_config_path=prompts_config_path)
        else:
            logger.warning(f"No specific reporter found for topic '{topic}'. Returning default reporter.")
            # Return an instance configured as default (identifier=None handles this)
            return cls(identifier=None, prompts_config_path=prompts_config_path)

    @classmethod
    def get_all_reporter_ids(cls, prompts_config_path: Path = PROMPTS_CONFIG_FILE) -> list[str]:
        """Gets a list of all available reporter IDs from the config."""
        # Initialize with None to load all configs
        instance = cls(identifier=None, prompts_config_path=prompts_config_path)
        return [reporter.get('id') for reporter in instance.all_reporters_data if reporter.get('id')]

    @classmethod
    def get_all_reporter_names(cls, prompts_config_path: Path = PROMPTS_CONFIG_FILE) -> list[str]:
        """Gets a list of all available reporter names from the config."""
        # Initialize with None to load all configs
        instance = cls(identifier=None, prompts_config_path=prompts_config_path)
        return [reporter.get('name', 'Unnamed Reporter') for reporter in instance.all_reporters_data]


    @classmethod
    def get_random_reporter(cls, prompts_config_path: Path = PROMPTS_CONFIG_FILE) -> 'Reporter':
        """Gets a randomly selected reporter instance (identified by ID)."""
        # Initialize with None to load all configs
        instance = cls(identifier=None, prompts_config_path=prompts_config_path)
        if not instance.all_reporters_data:
            logger.error("No reporters available to choose from.")
            # Return a default-initialized instance which will likely log errors
            return cls(identifier=None, prompts_config_path=prompts_config_path)

        # Get IDs of reporters that have an ID defined
        available_ids = [r.get('id') for r in instance.all_reporters_data if r.get('id')]
        if not available_ids:
             logger.error("Could not select a random reporter (no valid IDs found in config).")
             return cls(identifier=None, prompts_config_path=prompts_config_path)

        selected_id = random.choice(available_ids)
        logger.info(f"Randomly selected reporter ID: {selected_id}")
        return cls(identifier=selected_id, prompts_config_path=prompts_config_path)


# Example usage when run directly
if __name__ == "__main__":
    print("Available Reporter IDs:")
    ids = Reporter.get_all_reporter_ids()
    for rid in ids:
        print(f"- {rid}")

    print("\nTesting Default Reporter:")
    default_reporter = Reporter() # Uses identifier=None -> default logic
    print(f"  ID: {default_reporter.id}")
    print(f"  Name: {default_reporter.name}")
    print(f"  Topics: {default_reporter.topics}")
    print(f"  Model: {default_reporter.model}")
    print(f"  Voice: {default_reporter.voice}")

    print(f"  Prompt Preview: {default_reporter.prompt[:300]}...") # Uncomment to see prompt start

    print("\nTesting Reporter for Topic (tech):")
    tech_reporter = Reporter.get_reporter_for_topic("tech")
    print(f"  Selected Reporter ID: {tech_reporter.id}")
    print(f"  Selected Reporter Name: {tech_reporter.name}")
    print(f"  Model: {tech_reporter.model}")

    print("\nTesting Random Reporter:")
    random_reporter = Reporter.get_random_reporter()
    print(f"  Selected Reporter ID: {random_reporter.id}")
    print(f"  Selected Reporter Name: {random_reporter.name}")
    print(f"  Tags: {random_reporter.tags}")
