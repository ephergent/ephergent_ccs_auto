#!/usr/bin/env python3
import os
import logging
import shutil
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple # Added Tuple
from dotenv import load_dotenv

# Try to import ChromaDB and SentenceTransformer, handle import errors
try:
    import chromadb
    from sentence_transformers import SentenceTransformer
    VECTOR_DB_AVAILABLE = True
except ImportError:
    print("Warning: chromadb or sentence-transformers not found. Vector DB features will be disabled.")
    VECTOR_DB_AVAILABLE = False
    # Define dummy classes/functions if libraries are not installed
    class DummyChromaClient:
        def get_or_create_collection(self, name): return DummyChromaCollection()
        def heartbeat(self): return True
    class DummyChromaCollection:
        def add(self, documents, metadatas, ids): pass
        def query(self, query_texts, n_results): return {"ids": [[]], "documents": [[]], "metadatas": [[]]}
    class DummySentenceTransformer:
        def encode(self, text): return [0.0] * 768 # Return dummy embedding
    chromadb = type('chromadb', (object,), {'PersistentClient': DummyChromaClient})()
    SentenceTransformer = DummySentenceTransformer

# Local imports (assuming utils is in the same parent directory or PYTHONPATH is set correctly)
try:
    from utils.metadata_utils import extract_pelican_metadata # Import the utility function
except ImportError as e:
    print(f"Error: Could not import metadata_utils: {e}. Metadata extraction for archiving may fail.")
    # Define a dummy function if import fails
    def extract_pelican_metadata(markdown_content: str) -> Tuple[Dict[str, Any], str]:
        logger = logging.getLogger(__name__) # Need logger inside dummy
        logger.warning("Using dummy extract_pelican_metadata function.")
        # Return empty metadata and original content as fallback
        return {}, markdown_content


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# --- Configuration ---
ARCHIVE_ARTIFACTS = os.getenv('ARCHIVE_ARTIFACTS', 'true').lower() in ('true', '1', 'yes', 't')
BASE_ARCHIVE_DIR = Path(__file__).parent.parent / 'archive'
# Ensure the base archive directory exists
BASE_ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

# Vector DB Configuration
CHROMA_DB_PATH = Path(__file__).parent.parent / 'ephergent_db'
EMBEDDING_MODEL_NAME = os.getenv('EMBEDDING_MODEL_NAME', 'all-mpnet-base-v2') # Use env var or default
COLLECTION_NAME = "article_archive"

class Archiver:
    def __init__(self):
        """Initializes the Archiver with ChromaDB client and embedding model."""
        self.client = None
        self.collection = None
        self.embedding_model = None
        self.initialized = False

        if not VECTOR_DB_AVAILABLE:
            logger.warning("Vector DB libraries not available. Archiver initialized without vector DB support.")
            return

        logger.info(f"Initializing Archiver with ChromaDB path: {CHROMA_DB_PATH}")
        try:
            # Ensure the directory exists
            CHROMA_DB_PATH.mkdir(parents=True, exist_ok=True)
            self.client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))

            # Optional: Check if ChromaDB is running/accessible (heartbeat)
            try:
                self.client.heartbeat()
                logger.info("ChromaDB client connected successfully.")
            except Exception as e:
                logger.error(f"ChromaDB heartbeat failed: {e}. Ensure ChromaDB is running or path is correct.")
                self.client = None # Disable if heartbeat fails
                return # Exit init if client fails

            self.collection = self.client.get_or_create_collection(name=COLLECTION_NAME)
            logger.info(f"ChromaDB collection '{COLLECTION_NAME}' ready.")

            self.embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
            logger.info(f"Embedding model '{EMBEDDING_MODEL_NAME}' loaded.")

            self.initialized = True
            logger.info("Archiver initialized successfully with vector DB support.")

        except Exception as e:
            logger.error(f"Error initializing Archiver or Vector DB: {e}", exc_info=True)
            self.initialized = False # Ensure initialized is False on error


    def embed_text(self, text: str) -> Optional[list[float]]:
        """Generates an embedding for the given text."""
        if not self.initialized or not self.embedding_model:
            logger.warning("Archiver or embedding model not initialized. Cannot embed text.")
            return None
        if not text or not isinstance(text, str):
            logger.warning("Cannot embed empty or non-string text.")
            return None
        try:
            # Ensure text is not excessively long for the model if needed
            # (all-mpnet-base-v2 has a max sequence length of 384 tokens)
            # Simple truncation as a safeguard, more sophisticated methods exist
            max_length = 384 * 4 # Estimate tokens to chars (rough)
            if len(text) > max_length:
                 logger.warning(f"Text length ({len(text)}) exceeds typical model limit. Truncating for embedding.")
                 text = text[:max_length]

            embedding = self.embedding_model.encode(text).tolist()
            # logger.debug(f"Generated embedding for text (first 10 dims): {embedding[:10]}...")
            return embedding
        except Exception as e:
            logger.error(f"Error generating embedding for text: {e}", exc_info=True)
            return None

    def add_to_vector_db(self, article_id: str, text_content: str, metadata: dict):
        """Adds article content and metadata to the ChromaDB collection."""
        if not self.initialized or not self.collection:
            logger.warning("Archiver or ChromaDB collection not initialized. Cannot add to vector DB.")
            return

        if not article_id or not text_content:
            logger.warning("Article ID or text content is missing. Cannot add to vector DB.")
            return

        embedding = self.embed_text(text_content)
        if embedding is None:
            logger.error(f"Failed to generate embedding for article ID {article_id}. Cannot add to vector DB.")
            return

        # Ensure metadata is JSON serializable (ChromaDB requirement)
        # Convert Path objects to strings, handle lists/dicts recursively if needed
        cleaned_metadata = {}
        for key, value in metadata.items():
            try:
                # Attempt to JSON dump and load to check serializability
                json.dumps(value)
                cleaned_metadata[key] = value
            except (TypeError, OverflowError):
                logger.warning(f"Metadata key '{key}' for article ID {article_id} is not JSON serializable ({type(value)}). Converting to string.")
                cleaned_metadata[key] = str(value)
            except Exception as e:
                 logger.warning(f"Unexpected error checking serializability for metadata key '{key}': {e}. Converting to string.")
                 cleaned_metadata[key] = str(value)


        try:
            self.collection.add(
                documents=[text_content],
                embeddings=[embedding],
                metadatas=[cleaned_metadata],
                ids=[article_id]
            )
            logger.info(f"Successfully added article ID {article_id} to vector database.")
        except Exception as e:
            logger.error(f"Failed to add article ID {article_id} to vector database: {e}", exc_info=True)

    def search_similar_articles(self, query_text: str, top_k: int = 5) -> list[dict]:
        """
        Searches the vector database for articles similar to the query text.

        Args:
            query_text (str): The text to query for similarity.
            top_k (int): The number of top results to return.

        Returns:
            list[dict]: A list of dictionaries, where each dictionary contains
                        'id', 'document' (text content), and 'metadata' for a similar article.
                        Returns an empty list if search fails or no results.
        """
        if not self.initialized or not self.collection:
            logger.warning("Archiver or ChromaDB collection not initialized. Cannot search vector DB.")
            return []

        if not query_text or not isinstance(query_text, str):
            logger.warning("Query text is missing or not a string. Cannot search vector DB.")
            return []

        query_embedding = self.embed_text(query_text)
        if query_embedding is None:
            logger.error("Failed to generate embedding for query text. Cannot search vector DB.")
            return []

        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=['documents', 'metadatas'] # Include document text and metadata
            )
            logger.info(f"Vector database search completed for query.")

            # Format results
            formatted_results = []
            if results and results.get('ids') and results['ids'][0]:
                for i in range(len(results['ids'][0])):
                    article_id = results['ids'][0][i]
                    document = results['documents'][0][i]
                    metadata = results['metadatas'][0][i]
                    formatted_results.append({
                        'id': article_id,
                        'document': document,
                        'metadata': metadata
                    })
                logger.info(f"Found {len(formatted_results)} results.")
            else:
                logger.info("No results found for the query.")

            return formatted_results

        except Exception as e:
            logger.error(f"Error searching vector database: {e}", exc_info=True)
            return []


    def archive_artifacts(
        self,
        filename_base: str,
        artifact_paths: Dict[str, Path | List[Path] | None] # Accept list for images
        ) -> Path | None:
        """
        Archives generated artifacts into a timestamped subdirectory.
        Also adds the article content to the vector database if initialized.

        Args:
            filename_base (str): A base name derived from the article title and date,
                                 used for the archive subdirectory name.
            artifact_paths (Dict[str, Path | List[Path] | None]): A dictionary mapping artifact types
                                 (e.g., 'markdown', 'feature_image', 'article_images', 'audio', 'video', 'summary_file', 'image_prompts_file')
                                 to their paths in the temporary output directory.
                                 Values can be None or lists if an artifact wasn't generated or has multiple files.

        Returns:
            Path | None: The path to the created archive subdirectory, or None if archiving failed.
        """
        if not ARCHIVE_ARTIFACTS:
            logger.info("Archiving is disabled (ARCHIVE_ARTIFACTS is not 'true'). Skipping.")
            return None

        if not filename_base:
            logger.error("Filename base is required for archiving.")
            return None

        # Create the archive subdirectory name
        # Use the provided filename_base directly, assuming it already includes date/slug
        archive_subdir_name = filename_base
        archive_subdir = BASE_ARCHIVE_DIR / archive_subdir_name

        logger.info(f"Starting archiving process for: {filename_base}")
        logger.info(f"Target archive directory: {archive_subdir}")

        # Ensure the base archive directory exists (already done in __init__, but double check)
        BASE_ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

        try:
            # Create the archive subdirectory
            archive_subdir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created archive subdirectory: {archive_subdir}")

            # Helper function to copy artifacts
            def copy_artifact(source: Path | List[Path] | None, target_dir: Path, artifact_type: str) -> bool:
                copied_any = False
                if source is None:
                    logger.debug(f"No {artifact_type} artifact provided for archiving.")
                    return True # Nothing to copy is not a failure

                if isinstance(source, Path):
                    source_list = [source]
                elif isinstance(source, list):
                    source_list = source
                else:
                    logger.error(f"Invalid source type for {artifact_type}: {type(source)}")
                    return False # Invalid source is a failure

                target_dir.mkdir(parents=True, exist_ok=True) # Ensure target subdir exists

                for src_path in source_list:
                    if src_path and src_path.is_file():
                        target_path = target_dir / src_path.name
                        try:
                            shutil.copy2(src_path, target_path)
                            logger.info(f"Archived {artifact_type}: {target_path.name}")
                            copied_any = True
                        except Exception as e:
                            logger.error(f"Failed to copy {artifact_type} from {src_path} to {target_dir}: {e}")
                            # Continue trying other files in the list, but mark overall failure
                            copied_any = False # Mark failure if any copy fails
                    elif src_path:
                        logger.warning(f"{artifact_type.capitalize()} source path not found or is not a file: {src_path}")
                        # Treat missing optional files as non-fatal, but log
                        if artifact_type in ['markdown', 'summary_file']: # Essential files
                             copied_any = False # Mark failure if essential file is missing
                        else:
                             copied_any = True # Treat missing optional files as successful copy attempt

                return copied_any # Return True if at least one file was copied successfully or none were provided/required

            # Copy Markdown file (essential)
            markdown_path = artifact_paths.get("markdown")
            if not copy_artifact(markdown_path, archive_subdir, "markdown"):
                 logger.error("Failed to archive markdown file. Aborting archiving.")
                 # Clean up the partially created archive directory? Or leave for inspection?
                 # Leaving for inspection for now.
                 return None

            # Copy other artifacts to appropriate subdirectories within the archive
            copy_artifact(artifact_paths.get("feature_image"), archive_subdir / "images", "feature image")
            copy_artifact(artifact_paths.get("article_images"), archive_subdir / "images", "article images") # Handles list
            copy_artifact(artifact_paths.get("audio"), archive_subdir / "audio", "audio")
            copy_artifact(artifact_paths.get("video"), archive_subdir / "video", "video")
            copy_artifact(artifact_paths.get("summary_file"), archive_subdir, "summary file") # Summary in base archive dir
            copy_artifact(artifact_paths.get("image_prompts_file"), archive_subdir, "image prompts file") # Prompts in base archive dir


            # --- Add to Vector Database ---
            # Only add the main article markdown content to the vector DB
            article_text_for_db = None
            extracted_metadata_dict = {} # Initialize dictionary

            if markdown_path and markdown_path.is_file():
                try:
                    with open(markdown_path, 'r', encoding='utf-8') as f:
                        full_markdown_content = f.read()
                    # Use the utility function to get metadata and content body
                    extracted_metadata_dict, article_text_for_db = extract_pelican_metadata(full_markdown_content)
                    logger.info(f"Successfully read markdown content from {markdown_path} for vector DB.")
                except Exception as e:
                    logger.error(f"Failed to read or parse markdown content from {markdown_path} for vector DB: {e}", exc_info=True)
                    article_text_for_db = None # Ensure it's None if reading/parsing fails

            if article_text_for_db and self.initialized:
                # Prepare metadata for the vector DB entry
                # Use the extracted metadata, add filename_base and archive_subdir_name
                db_metadata = extracted_metadata_dict.copy()
                db_metadata['filename_base'] = filename_base
                db_metadata['archive_subdir'] = str(archive_subdir.relative_to(BASE_ARCHIVE_DIR)) # Store relative path

                # Ensure 'tags' is stored as a JSON string in metadata for ChromaDB
                # extract_pelican_metadata already handles this, but double-check
                if 'tags' in db_metadata and isinstance(db_metadata['tags'], list):
                     db_metadata['tags'] = json.dumps(db_metadata['tags'])
                     logger.debug("Converted tags list to JSON string for vector DB metadata.")
                elif 'tags' not in db_metadata:
                     db_metadata['tags'] = json.dumps([]) # Ensure tags key exists, even if empty

                # Add the article to the vector database
                self.add_to_vector_db(
                    article_id=filename_base, # Use filename_base as the unique ID
                    text_content=article_text_for_db,
                    metadata=db_metadata
                )
            elif not article_text_for_db:
                 logger.warning(f"No article text available from {markdown_path} to add to vector database.")
            else: # self.initialized is False
                 logger.info("Vector database not initialized. Skipping adding article content.")


            logger.info(f"Archiving and vector DB processing completed for {filename_base}.")
            return archive_subdir # Return the path to the created archive directory

        except Exception as e:
            logger.error(f"An unexpected error occurred during archiving for {filename_base}: {e}", exc_info=True)
            # Decide if partial archive should be cleaned up or left
            # Leaving it for now for debugging
            return None


# Example usage when run directly
if __name__ == "__main__":
    print("--- Running Archiver Standalone Test ---")

    # Create a dummy output directory for the test run
    test_output_dir = Path(__file__).parent.parent / 'output' / 'archiver_test'
    test_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Using test output directory: {test_output_dir}")

    # Create dummy artifacts for testing
    dummy_images_dir = test_output_dir / "images"
    dummy_audio_dir = test_output_dir / "audio"
    dummy_images_dir.mkdir(exist_ok=True)
    dummy_audio_dir.mkdir(exist_ok=True)

    test_filename_base = f"{datetime.now().strftime('%Y%m%d')}_test_article"

    dummy_markdown_content = f"""Title: Test Article for Archiving
Date: 2025-05-17 10:00:00
Category: Test
Tags: testing, archiving, vector_db
Slug: {test_filename_base}
Author: Test Reporter
Summary: This is a test summary.
Status: draft
Stardate: 100.100.100
Location: Test Location

This is the main content of the test article.
It has multiple paragraphs.

This content should be archived and added to the vector database.
It mentions keywords like 'vector database', 'archiving', and 'Pelican'.
"""
    dummy_markdown_path = test_output_dir / f"{test_filename_base}.md"
    with open(dummy_markdown_path, "w", encoding="utf-8") as f:
        f.write(dummy_markdown_content)
    print(f"Created dummy markdown: {dummy_markdown_path}")

    dummy_feature_img = dummy_images_dir / "test_feature.png"
    dummy_article_img1 = dummy_images_dir / "test_article_essence.jpg"
    dummy_audio = dummy_audio_dir / "test_audio_combined.mp3"
    dummy_summary = test_output_dir / f"{test_filename_base}_summary.txt"
    dummy_prompts_json = test_output_dir / f"{test_filename_base}_image_prompts.json"

    try:
        dummy_feature_img.touch()
        dummy_article_img1.touch()
        dummy_audio.touch()
        dummy_summary.write_text("This is a dummy summary file.")
        dummy_prompts_json.write_text('{"featured": "dummy prompt"}')
        print("Created dummy artifact files.")

        # Prepare artifact paths dictionary
        test_artifact_paths = {
            "markdown": dummy_markdown_path,
            "feature_image": dummy_feature_img,
            "article_images": [dummy_article_img1], # Pass as list
            "audio": dummy_audio,
            "video": None, # Simulate no video generated
            "summary_file": dummy_summary,
            "image_prompts_file": dummy_prompts_json
        }

        # Initialize Archiver
        archiver = Archiver()

        # Run archiving process
        archive_result_path = archiver.archive_artifacts(test_filename_base, test_artifact_paths)

        if archive_result_path:
            print(f"\nArchiving successful! Artifacts archived to: {archive_result_path}")
            # Verify files exist in archive
            print("Verifying archived files:")
            print(f"- Markdown: {archive_result_path / dummy_markdown_path.name} exists: {(archive_result_path / dummy_markdown_path.name).exists()}")
            print(f"- Feature Image: {archive_result_path / 'images' / dummy_feature_img.name} exists: {(archive_result_path / 'images' / dummy_feature_img.name).exists()}")
            print(f"- Article Image: {archive_result_path / 'images' / dummy_article_img1.name} exists: {(archive_result_path / 'images' / dummy_article_img1.name).exists()}")
            print(f"- Audio: {archive_result_path / 'audio' / dummy_audio.name} exists: {(archive_result_path / 'audio' / dummy_audio.name).exists()}")
            print(f"- Summary: {archive_result_path / dummy_summary.name} exists: {(archive_result_path / dummy_summary.name).exists()}")
            print(f"- Prompts JSON: {archive_result_path / dummy_prompts_json.name} exists: {(archive_result_path / dummy_prompts_json.name).exists()}")

            # Test vector DB search (if initialized)
            if archiver.initialized:
                print("\nTesting Vector DB search...")
                search_query = "articles about databases or storage"
                search_results = archiver.search_similar_articles(search_query, top_k=3)
                print(f"Search query: '{search_query}'")
                if search_results:
                    print(f"Found {len(search_results)} similar articles:")
                    for res in search_results:
                        print(f"  - ID: {res['id']}, Title: {res['metadata'].get('title', 'N/A')}")
                else:
                    print("No similar articles found in the vector database.")
            else:
                print("\nVector DB not initialized. Skipping search test.")

    except Exception as e:
        print(f"\nAn unexpected error occurred during the test: {e}")
        logger.error("Test run failed.", exc_info=True)
    finally:
        # Clean up dummy files/dir
        if test_output_dir.exists():
            print("\nCleaning up temporary test output directory...")
            shutil.rmtree(test_output_dir)
            print("Cleanup complete.")
        # Note: The actual archive directory created by the test is NOT cleaned up automatically.
        # It's located at BASE_ARCHIVE_DIR / test_filename_base
        print(f"Test archive directory is located at: {BASE_ARCHIVE_DIR / test_filename_base}")
