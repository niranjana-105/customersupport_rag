FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# Install the project into `/app`
WORKDIR /app

ENV UV_COMPILE_BYTECODE=1

# Copy from the cache instead of linking since it's a mounted volume
ENV UV_LINK_MODE=copy

# Copy dependency files first for optimal layer caching
COPY pyproject.toml requirements.txt ./

# Install the project's dependencies using pip (no uv.lock required)
RUN pip install --no-cache-dir -r requirements.txt

# Then, add the rest of the project source code
COPY . /app

# Place executables in the environment at the front of the path
ENV PATH="/app/.venv/bin:$PATH"

# Reset the entrypoint, don't invoke `uv`
ENTRYPOINT []

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
