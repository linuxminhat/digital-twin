# syntax=docker/dockerfile:1
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ src/
COPY evaluation/ evaluation/
COPY knowledge-base/ knowledge-base/
COPY prompts/ prompts/

RUN useradd --create-home --uid 1000 twin && chown -R twin /app
USER twin

# Fail the build if the knowledge base no longer parses or the benchmark regresses.
RUN python -m evaluation.eval_runner --min-recall 0.9 --min-rejection 1.0 --max-false-refusal 0.1 > /dev/null

ENTRYPOINT ["python", "-m", "src.cli"]
CMD ["--help"]
