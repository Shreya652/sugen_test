FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY requirements_test04.txt .
RUN pip install -r requirements_test04.txt && mkdir -p /in /out

COPY score.py llm.py test03_review.py test04_agents.py ./
COPY prompts/ ./prompts/
COPY data/mms-2026-01.json data/final-icd-codes.csv data/dev_notes.jsonl ./data/
COPY run/ ./run/

ENTRYPOINT ["python3", "/app/run/coder.py"]
