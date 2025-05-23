#!/usr/bin/env python3
import os
import logging
import subprocess
from pathlib import Path
from datetime import datetime # Import datetime
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# --- Configuration ---
PELICAN_PROJECT_DIR = os.getenv('PELICAN_PROJECT_DIR')
PUSH_TO_GIT = os.getenv('PUSH_TO_GIT', 'false').lower() in ('true', '1', 'yes', 't')
GIT_BRANCH = os.getenv('GIT_BRANCH', 'main') # Allow configuring the branch

def run_git_command(command: list[str], working_dir: Path) -> bool:
    """Runs a Git command in a specified directory and logs the output."""
    try:
        logger.info(f"Running command: {' '.join(command)} in {working_dir}")
        # Using capture_output=True to get stdout/stderr
        # Using text=True to decode stdout/stderr as text
        # Using check=True to raise CalledProcessError on non-zero exit codes
        result = subprocess.run(
            command,
            cwd=working_dir,
            check=True,
            capture_output=True,
            text=True,
            timeout=120 # Add a timeout (e.g., 2 minutes)
        )
        logger.info(f"Command successful. Output:\n{result.stdout}")
        if result.stderr:
             logger.warning(f"Command stderr:\n{result.stderr}")
        return True
    except FileNotFoundError:
        logger.error(f"Error: 'git' command not found. Is Git installed and in your PATH?")
        return False
    except subprocess.CalledProcessError as e:
        logger.error(f"Git command failed with exit code {e.returncode}: {' '.join(command)}")
        logger.error(f"Stderr:\n{e.stderr}")
        logger.error(f"Stdout:\n{e.stdout}")
        return False
    except subprocess.TimeoutExpired:
        logger.error(f"Git command timed out: {' '.join(command)}")
        return False
    except Exception as e:
        logger.error(f"An unexpected error occurred while running git command: {e}", exc_info=True)
        return False


def publish_to_git(commit_message: str) -> bool:
    """
    Adds all changes, commits, and pushes them to the configured Git repository
    within the PELICAN_PROJECT_DIR.

    Args:
        commit_message (str): The commit message to use.

    Returns:
        bool: True if all Git operations were successful, False otherwise.
    """
    if not PUSH_TO_GIT:
        logger.info("Git publishing is disabled (PUSH_TO_GIT is not 'true'). Skipping.")
        return False # Return False to indicate it didn't run

    if not PELICAN_PROJECT_DIR:
        logger.error("PELICAN_PROJECT_DIR environment variable not set. Cannot publish to Git.")
        return False

    project_path = Path(PELICAN_PROJECT_DIR)
    if not project_path.is_dir():
        logger.error(f"PELICAN_PROJECT_DIR '{PELICAN_PROJECT_DIR}' does not exist or is not a directory.")
        return False

    # Check if it's a git repository
    git_dir = project_path / ".git"
    if not git_dir.is_dir():
        logger.error(f"'{PELICAN_PROJECT_DIR}' does not appear to be a Git repository (missing .git directory).")
        return False

    logger.info(f"--- Starting Git publication process for {project_path} ---")

    # 1. Git Add
    if not run_git_command(['git', 'add', '.'], working_dir=project_path):
        logger.error("Git add command failed.")
        return False

    # 2. Git Commit
    # Check if there are changes to commit first
    status_result = subprocess.run(['git', 'status', '--porcelain'], cwd=project_path, capture_output=True, text=True)
    if not status_result.stdout.strip():
        logger.info("No changes detected to commit.")
        # Decide if this is success or failure - arguably success as repo is up-to-date
        # However, if we *expect* changes, maybe it's a warning?
        # Let's consider it success for now, but maybe return a specific status later.
        # If we push anyway, it might just say 'Everything up-to-date'.
        # Let's proceed to push.
        pass # Continue to push step

    elif not run_git_command(['git', 'commit', '-m', commit_message], working_dir=project_path):
        logger.error("Git commit command failed.")
        # Check if commit failed because there was nothing to commit (though status check should prevent this)
        if "nothing to commit" in status_result.stdout.lower():
             logger.info("Commit failed because there were no changes staged (redundant check).")
             # Continue to push
        else:
             return False


    # 3. Git Push
    if not run_git_command(['git', 'push', 'origin', GIT_BRANCH], working_dir=project_path):
        logger.error(f"Git push to origin/{GIT_BRANCH} command failed.")
        return False

    logger.info(f"--- Git publication process completed successfully for {project_path} ---")
    return True


# Example usage when run directly
if __name__ == "__main__":
    print("Testing Git Publisher...")

    if not PUSH_TO_GIT:
        print("PUSH_TO_GIT is not enabled in .env. Skipping actual Git commands.")
        print("Set PUSH_TO_GIT=true and configure PELICAN_PROJECT_DIR to run the test.")
    elif not PELICAN_PROJECT_DIR or not Path(PELICAN_PROJECT_DIR).is_dir():
        print("PELICAN_PROJECT_DIR is not set or is not a valid directory in .env. Skipping test.")
    else:
        print(f"Attempting to add, commit, and push to: {PELICAN_PROJECT_DIR}")
        # Create a dummy file change for testing purposes
        dummy_file = Path(PELICAN_PROJECT_DIR) / "git_publish_test.tmp"
        try:
            with open(dummy_file, "w") as f:
                f.write(f"Test change at {datetime.now()}\n")
            print(f"Created/modified dummy file: {dummy_file}")

            test_commit_message = f"Test commit via git_publisher.py at {datetime.now()}"
            success = publish_to_git(test_commit_message)

            if success:
                print("\nGit publish test completed successfully (check your repository).")
            else:
                print("\nGit publish test failed.")

        except Exception as e:
            print(f"\nAn error occurred during the test: {e}")
        finally:
            # Clean up the dummy file
            if dummy_file.exists():
                try:
                    dummy_file.unlink()
                    print(f"Removed dummy file: {dummy_file}")
                    # Optionally, commit the removal if the initial push succeeded
                    # run_git_command(['git', 'add', dummy_file.name], working_dir=Path(PELICAN_PROJECT_DIR))
                    # run_git_command(['git', 'commit', '-m', 'Clean up test file'], working_dir=Path(PELICAN_PROJECT_DIR))
                    # run_git_command(['git', 'push', 'origin', GIT_BRANCH], working_dir=Path(PELICAN_PROJECT_DIR))
                except Exception as e:
                    print(f"Warning: Failed to clean up dummy file {dummy_file}: {e}")
