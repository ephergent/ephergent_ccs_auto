# TODO

GOAL: Ideally we have Pixel, A1, Clive, and one Denizen all published on the same day.

## Plan to achieve "Same Day" Publishing:

This plan involves modifying `app.py` to accept explicit month and week arguments, and then updating `generate_articles_all_reporters.py` to pass these arguments for all generated content, including a Denizen profile.

### Phase 1: Modify `app.py` to accept a specific publish date. (COMPLETED)

### Phase 2: Modify `generate_articles_all_reporters.py` to use the new date arguments. (COMPLETED)

### Phase 3: Automate content generation from `ephergent_new_stories.json` for multiple weeks.

This phase will refactor `generate_articles_all_reporters.py` to iterate through predefined story ideas, assigning each to a sequential week and generating all associated articles (reporters + Denizen) for that week.

1.  **Update `generate_articles_all_reporters.py` to read new story ideas:**
    *   Add a new constant `EPH_NEW_STORIES_FILE` pointing to `input/ephergent_new_stories.json`.
    *   Load the `ephergent_story_ideas` from this JSON file within the `generate_articles_for_all_reporters` function.
    *   **Code Change (generate_articles_all_reporters.py):**
        ```python
        # Add new constant near existing file paths (e.g., after APP_SCRIPT):
        EPH_NEW_STORIES_FILE = Path(__file__).parent / 'input' / 'ephergent_new_stories.json'

        # Inside generate_articles_for_all_reporters function, after checking APP_SCRIPT exists:
        if not EPH_NEW_STORIES_FILE.exists():
            logger.error(f"Error: New stories file not found at {EPH_NEW_STORIES_FILE}")
            sys.exit(1)

        try:
            with open(EPH_NEW_STORIES_FILE, 'r', encoding='utf-8') as f:
                new_stories_data = json.load(f)
            story_ideas = new_stories_data.get('ephergent_story_ideas', [])
            if not story_ideas:
                logger.warning("No story ideas found in ephergent_new_stories.json. Exiting.")
                sys.exit(0)
            logger.info(f"Found {len(story_ideas)} new story ideas to process.")
        except json.JSONDecodeError:
            logger.error(f"Error: Could not parse JSON from {EPH_NEW_STORIES_FILE}. Please check the file format.")
            sys.exit(1)
        ```

2.  **Modify `generate_articles_for_all_reporters` function signature and loop structure:**
    *   Change the parameters to `start_month`, `start_week`, `generate_denizen`, `start_index`.
    *   Introduce an outer loop that iterates through `story_ideas`.
    *   Inside this loop, calculate the `current_month` and `current_week` for each story idea.
    *   Construct the `topic_string` for each story idea using its `title`, `hook`, and `character_focus` details.
    *   Reset `current_article_index_for_week` to `start_index` for each new week.
    *   Pass `current_month`, `current_week`, and `current_article_index_for_week` to the `app.py` calls for both reporters and the Denizen.
    *   **Code Change (generate_articles_all_reporters.py):**
        ```python
        # Update function signature:
        def generate_articles_for_all_reporters(start_month: int, start_week: int, generate_denizen: bool, start_index: int):
            # ... (existing code for loading reporters and skip_steps) ...

            current_month = start_month
            current_week = start_week

            # Remove the old `topic_string = """ """` line, as it will be generated per story idea

            for i, story_idea in enumerate(story_ideas):
                # For subsequent stories, increment week and potentially month
                if i > 0: # Only increment for the 2nd story idea onwards
                    current_week += 1
                    if current_week > 4: # Assuming 4 weeks per month
                        current_week = 1
                        current_month += 1

                logger.info(f"\n--- Processing Story Idea {i+1}: '{story_idea['title']}' for Cycle {current_month:03d}, Week {current_week:02d} ---")

                # Construct the topic string for the LLM
                topic_parts = [
                    story_idea['title'],
                    story_idea['hook']
                ]
                char_focus_str = []
                # Iterate through character_focus to add details to the topic
                for char_id, focus_desc in story_idea['character_focus'].items():
                    # Find reporter name from loaded reporters data for better topic string
                    reporter_name = next((r['name'] for r in reporters if r['id'] == char_id), char_id)
                    char_focus_str.append(f"{reporter_name}: {focus_desc}")
                if char_focus_str:
                    topic_parts.append("Character Focus: " + "; ".join(char_focus_str))

                topic_string = " ".join(topic_parts).strip()
                logger.info(f"Constructed topic: {topic_string[:200]}...") # Log a snippet

                current_article_index_for_week = start_index # Reset index for each new week

                for reporter in reporters:
                    reporter_id = reporter.get('id')
                    reporter_name = reporter.get('name', reporter_id)

                    if not reporter_id:
                        logger.warning(f"Warning: Skipping reporter entry with no ID: {reporter}")
                        continue

                    logger.info(f"\n--- Generating article for reporter: {reporter_name} (ID: {reporter_id}) ---")

                    command = [
                        sys.executable,
                        str(APP_SCRIPT),
                        "--auto-generate",
                        "--reporter", reporter_id,
                        "--topic", topic_string, # Use the dynamically generated topic
                        "--skip", skip_steps,
                        "--month", str(current_month), # Pass current month
                        "--week", str(current_week),   # Pass current week
                        "--index", str(current_article_index_for_week) # Pass current index
                    ]

                    logger.info(f"Running command: {' '.join(command)}")

                    try:
                        result = subprocess.run(command, capture_output=True, text=True, check=True)
                        logger.info(f"Successfully generated article for {reporter_name}.")
                    except subprocess.CalledProcessError as e:
                        logger.error(f"Error generating article for {reporter_name} (ID: {reporter_id}): Command failed with exit code {e.returncode}")
                        logger.error(f"Subprocess stderr:\n{e.stderr}")
                        logger.error(f"Subprocess stdout:\n{e.stdout}")
                    except FileNotFoundError:
                        logger.error(f"Error: Could not find the Python executable '{sys.executable}' or the script '{APP_SCRIPT}'.")
                        logger.error("Please ensure Python is in your PATH and app.py exists.")
                        sys.exit(1)
                    except Exception as e:
                        logger.error(f"An unexpected error occurred while processing {reporter_name} (ID: {reporter_id}): {e}", exc_info=True)

                    current_article_index_for_week += 1 # Increment for the next article in this week

                # Denizen generation for the current week
                if generate_denizen:
                    logger.info("\n--- Generating Dimensional Denizen profile ---")
                    denizen_command = [
                        sys.executable,
                        str(APP_SCRIPT),
                        "--denizen",
                        "--skip", skip_steps, # Apply skip steps to denizen generation too
                        "--month", str(current_month), # Pass current month
                        "--week", str(current_week),   # Pass current week
                        "--index", str(current_article_index_for_week) # Pass current index for Denizen
                    ]

                    logger.info(f"Running command: {' '.join(denizen_command)}")
                    try:
                        result = subprocess.run(denizen_command, capture_output=True, text=True, check=True)
                        logger.info("Successfully generated Dimensional Denizen profile.")
                    except subprocess.CalledProcessError as e:
                        logger.error(f"Error generating Denizen profile: Command failed with exit code {e.returncode}")
                        logger.error(f"Subprocess stderr:\n{e.stderr}")
                        logger.error(f"Subprocess stdout:\n{e.stdout}")
                    except Exception as e:
                        logger.error(f"An unexpected error occurred while generating Denizen profile: {e}", exc_info=True)

                current_article_index_for_week += 1 # Increment for the next article (if any) or next week's start

            logger.info("\n--- Finished generating articles for all reporters and Denizen for all story ideas ---")
        ```

3.  **Update `generate_articles_all_reporters.py`'s `ArgumentParser`:**
    *   Change `--month` to `--start-month` and `--week` to `--start-week`.
    *   Adjust default values as requested (start with Month 1, Week 2).
    *   Remove `--topic` and `--reporter` arguments from the `ArgumentParser` as they are now handled dynamically by the loop.
    *   **Code Change (generate_articles_all_reporters.py):**
        ```python
        if __name__ == "__main__":
            parser = argparse.ArgumentParser(description="Generate articles for all reporters and optionally a Denizen based on new story ideas.")
            # Remove --month, --week, --reporter, --topic arguments
            parser.add_argument(
                "--start-month",
                type=int,
                default=1, # Default to 1 for Cycle 001
                help="Specify the starting month number for the first generated story (e.g., 1 for Cycle 001)."
            )
            parser.add_argument(
                "--start-week",
                type=int,
                default=2, # Default to 2 for Week 02, as Week 01 is done
                help="Specify the starting week number for the first generated story (e.g., 2 for Week 02)."
            )
            parser.add_argument(
                "--denizen",
                action="store_true",
                help="Also generate a Dimensional Denizen profile for each week."
            )
            parser.add_argument(
                "--start-index",
                type=int,
                default=1, # Default to 1 for the first article of each week
                help="Specify the starting sub-index for the first article of each week (e.g., 1 for 001_01-01.json)."
            )
            args = parser.parse_args()

            # Basic validation for month/week
            if args.start_month is None or args.start_week is None: # Make them required now
                parser.error("Both --start-month and --start-week must be provided.")

            generate_articles_for_all_reporters(args.start_month, args.start_week, args.denizen, args.start_index)
        ```
