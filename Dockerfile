# Using Docker image with UV following the documentation at https://docs.astral.sh/uv/guides/integration/docker/

# Use a recommended Python image with uv pre-installed
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# Set the working directory
WORKDIR /app

# Copy project configuration and lock file
COPY pyproject.toml uv.lock ./

# Create the virtual environment and install all dependencies from the lockfile.
# This single command creates the .venv and installs everything needed.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked

# Copy the rest of your application source code
COPY . .

# Add the virtual environment's bin directory to the PATH.
ENV PATH="/app/.venv/bin:$PATH"

# Run the embedding task to prepare the application
RUN chmod +x /app/entrypoint.sh
ENTRYPOINT ["/app/entrypoint.sh"]

# Run streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]