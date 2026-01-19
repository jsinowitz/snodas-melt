FROM mambaorg/micromamba:1.5.8
ARG MAMBA_USER=mambauser
WORKDIR /app
COPY requirements.txt /app/
RUN micromamba create -y -n snodas -c conda-forge \
      python=3.11 \
      rasterio=1.3.10 \
      gdal \
      xarray=2024.3.0 \
      rioxarray=0.15.5 \
      cfgrib \
      eccodes \
      pyproj \
  && micromamba clean --all --yes
COPY requirements.txt /app/
RUN micromamba run -n snodas pip install -r requirements.txt \
  && micromamba clean --all --yes
USER root
COPY server.py /app/
COPY server_big.py /app/
COPY web /app/web
COPY start.sh /app/start.sh
RUN chmod 0755 /app/start.sh && \
    mkdir -p /app/cache /home/mambauser/.cache/mamba/proc && \
    chown -R mambauser:mambauser /app /home/mambauser
USER mambauser
ENV HOME=/home/mambauser
ENV XDG_CACHE_HOME=/home/mambauser/.cache
ENV MAMBA_NO_LOCK=1
ENV SNODAS_CACHE=/app/cache
EXPOSE 8000
CMD ["/app/start.sh"]
