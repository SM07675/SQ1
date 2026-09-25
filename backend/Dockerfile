FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    SATQUERY_ENV=production \
    SATQUERY_OFFLINE_MODE=false \
    SATQUERY_MODEL_DIR=/home/user/app/models \
    SATQUERY_ARTIFACT_DIR=/data/artifacts \
    SATQUERY_DATABASE_PATH=/data/artifacts/satquery.sqlite3 \
    HF_HOME=/home/user/.cache/huggingface

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -m -u 1000 user \
    && mkdir -p /data/artifacts /home/user/app \
    && chown -R user:user /data /home/user/app

USER user
WORKDIR /home/user/app
ENV PATH=/home/user/.local/bin:$PATH

COPY --chown=user pyproject.toml ./
COPY --chown=user app ./app
COPY --chown=user satquery_engine ./satquery_engine
RUN pip install --no-cache-dir --user torch==2.5.1+cpu torchvision==0.20.1+cpu --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir --user .

EXPOSE 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
