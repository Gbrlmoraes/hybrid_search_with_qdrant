#!/bin/sh
#
# This script runs on container startup.
set -e

echo "Loading embeddings into Qdrant..."
uv run task embed
echo "Embeddings loaded successfully."

exec "$@"