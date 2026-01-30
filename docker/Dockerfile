# FastSIMUS Development Environment
FROM python:3.12-slim-bookworm

# Copy uv binary from official image (recommended by Astral)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        curl \
        ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Install starship prompt
RUN curl -sS https://starship.rs/install.sh | sh -s -- --yes

# Install poethepoet via uv
RUN uv tool install poethepoet

# Copy starship configuration
COPY starship.toml /root/.config/starship.toml

# Set up bash to use starship
RUN echo 'eval "$(starship init bash)"' >> /root/.bashrc

# Set working directory
WORKDIR /workspace

# Default command
CMD ["/bin/bash"]
