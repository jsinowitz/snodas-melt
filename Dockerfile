FROM mambaorg/micromamba:1.5.8
WORKDIR /app
COPY requirements.txt /app/
RUN micromamba create -y -n snodas -c conda-forge python=3.11 && \
    micromamba run -n snodas pip install -r requirements.txt && \
    micromamba clean --all --yes
COPY server.py /app/
COPY web /app/web
ENV SNODAS_CACHE=/app/cache
EXPOSE 8000
CMD ["micromamba","run","-n","snodas","uvicorn","server:app","--host","0.0.0.0","--port","8000"]
