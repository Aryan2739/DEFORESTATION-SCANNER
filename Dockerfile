FROM python:3.11-slim

# Install C-level GDAL and geospatial dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gdal-bin \
    libgdal-dev \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user (required by Hugging Face Spaces)
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# Install locked Python dependencies
COPY --chown=user ./requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy project files and create writable cache/assets directories
COPY --chown=user . /app
RUN mkdir -p /app/.cache /app/assets

# Hugging Face Spaces routes traffic to port 7860 by default
EXPOSE 7860

CMD ["streamlit", "run", "ui.py", "--server.port=7860", "--server.address=0.0.0.0"]