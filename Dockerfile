FROM mambaorg/micromamba:1.5.8

# Micromamba image defaults
ARG MAMBA_USER=mambauser
WORKDIR /app

# Copy deps and create env
COPY requirements.txt /app/
# Create env with geo stack from conda-forge
RUN micromamba create -y -n snodas -c conda-forge \micromamba create -y -n snodas -c conda-forge \
      python=3.11 \
      rasterio=1.3.10 \
      gdal \
      xarray=2024.3.0 \
      rioxarray=0.15.5 \
      cfgrib \
      eccodes \
      pyproj \
  && micromamba clean --all --yes

# Then install light, pure-Python deps via pip into that env
COPY requirements.txt /app/
RUN micromamba run -n snodas pip install -r requirements.txt \
  && micromamba clean --all --yes



# Do file operations & perms as root first
USER root
COPY server.py /app/
COPY web /app/web
COPY start.sh /app/start.sh

# ensure dirs & ownership
RUN chmod 0755 /app/start.sh && \
    mkdir -p /app/cache /home/mambauser/.cache/mamba/proc && \
    chown -R mambauser:mambauser /app /home/mambauser

# drop back to non-root
USER mambauser

# make sure HOME and cache paths are writable
ENV HOME=/home/mambauser
ENV XDG_CACHE_HOME=/home/mambauser/.cache

# 👉 turn off micromamba locks at runtime
ENV MAMBA_NO_LOCK=1
# ENV CFGRIB_INDEXPATH=/app/cache/cfgrib

ENV SNODAS_CACHE=/app/cache
EXPOSE 8000

# 👉 also pass --no-lock explicitly at runtime
CMD ["/app/start.sh"]
