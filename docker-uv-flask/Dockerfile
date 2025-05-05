
FROM python:3.11-slim-bookworm
COPY --from=ghcr.io/astral-sh/uv:0.6.14 /uv /uvx /bin/

# EXPOSE 5000

# Copy the project into the image
ADD . /app

# Sync the project into a new environment, using the frozen lockfile
WORKDIR /app
RUN uv sync --frozen --no-dev --no-install-project

# Place executables in the environment at the front of the path
ENV PATH="/app/.venv/bin:$PATH"

# RUN uv run app_flask.py
CMD ["uv", "run", "app_flask.py"]


# https://medium.com/@albertazzir/blazing-fast-python-docker-builds-with-poetry-a78a66f5aed0

# https://github.com/astral-sh/uv-docker-example/blob/main/Dockerfile
# Use a Python image with uv pre-installed
# FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# # Install the project into `/app`
# WORKDIR /app

# # Enable bytecode compilation
# ENV UV_COMPILE_BYTECODE=1

# # Copy from the cache instead of linking since it's a mounted volume
# ENV UV_LINK_MODE=copy

# # Install the project's dependencies using the lockfile and settings
# RUN --mount=type=cache,target=/root/.cache/uv \
#     --mount=type=bind,source=uv.lock,target=uv.lock \
#     --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
#     uv sync --frozen --no-install-project --no-dev

# # Then, add the rest of the project source code and install it
# # Installing separately from its dependencies allows optimal layer caching
# ADD . /app
# RUN --mount=type=cache,target=/root/.cache/uv \
#     uv sync --frozen --no-dev

# # Place executables in the environment at the front of the path
# ENV PATH="/app/.venv/bin:$PATH"

# # Reset the entrypoint, don't invoke `uv`
# ENTRYPOINT []

# # Run the FastAPI application by default
# # Uses `fastapi dev` to enable hot-reloading when the `watch` sync occurs
# # Uses `--host 0.0.0.0` to allow access from outside the container
# CMD ["fastapi", "dev", "--host", "0.0.0.0", "src/uv_docker_example"]
