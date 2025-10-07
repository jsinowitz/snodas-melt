# === Base environment ===
FROM mambaorg/micromamba:1.5.8
WORKDIR /app
RUN micromamba create -y -n snodas -c conda-forge python=3.11 && \
# Copy dependency list first for caching
COPY requirements.txt /app/

# Install environment + Python libs
RUN micromamba create -y -n snodas -c conda-forge python=3.11 && \
    micromamba run -n snodas pip install -r requirements.txt && \
    micromamba clean --all --yes

# === Copy application code ===
COPY server.py /app/
COPY web /app/web
COPY start.sh /app/start.sh

# Make sure our start script is executable
RUN chmod +x /app/start.sh

# === Runtime configuration ===
ENV SNODAS_CACHE=/app/cache
EXPOSE 8000

# Run the warm-cache script on container start
CMD ["micromamba", "run", "-n", "snodas", "/app/start.sh"]

