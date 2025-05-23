#!/usr/bin/env python3
import logging
import re
import json # To handle tags list
from typing import Tuple, Dict, Any

logger = logging.getLogger(__name__)

def extract_pelican_metadata(markdown_content: str) -> Tuple[Dict[str, Any], str]:
    """
    Extracts metadata key-value pairs from a Pelican markdown header
    and separates the content body.

    Args:
        markdown_content: The full content of the markdown file as a string.

    Returns:
        A tuple containing:
        - A dictionary containing the extracted metadata. Keys are lowercased.
          Tags are expected to be comma-separated and returned as a JSON string list.
        - A string containing the content body following the metadata block.
          Returns ({}, markdown_content) if no metadata block is found.
    """
    metadata = {}
    lines = markdown_content.splitlines()
    header_pattern = re.compile(r'^([A-Za-z]+):\s*(.*)') # Simple key: value pattern
    metadata_end_index = -1 # Index of the line *after* the metadata block

    for i, line in enumerate(lines):
        stripped_line = line.strip()

        if stripped_line == "":
            # Found the blank line separator after metadata
            if metadata: # Only treat as separator if we've found metadata lines before it
                metadata_end_index = i + 1
                break
            # If no metadata found yet, ignore leading blank lines and continue
            continue

        match = header_pattern.match(line)
        if match:
            key = match.group(1).lower()
            value = match.group(2).strip()
            if key == 'tags':
                # Store tags as a JSON string list for ChromaDB compatibility
                tags_list = [tag.strip() for tag in value.split(',') if tag.strip()]
                metadata[key] = json.dumps(tags_list)
            else:
                metadata[key] = value
            # Keep track of the line after the last potential metadata line
            metadata_end_index = i + 1
        else:
            # Found a non-blank line that doesn't match the metadata pattern.
            # This line is the start of the content body.
            metadata_end_index = i
            break

    # If the loop finished without finding a break condition (e.g., no blank line,
    # or file ends immediately after metadata), metadata_end_index will be the
    # index after the last processed line, or -1 if no lines were processed.
    # The content body starts from metadata_end_index onwards.
    # If metadata_end_index is -1 (no metadata or blank lines found), content starts at 0.
    content_start_index = metadata_end_index if metadata_end_index != -1 else 0
    content_body = "\n".join(lines[content_start_index:])

    logger.debug(f"Extracted metadata: {metadata}")
    # logger.debug(f"Content body starts at line {content_start_index}, first 100 chars: {content_body[:100]}...") # Avoid logging huge content
    return metadata, content_body

