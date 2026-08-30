FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /opt/buildcheck-agent

COPY backend/requirements.txt backend/requirements.txt
RUN python -m pip install --no-cache-dir -r backend/requirements.txt

COPY . .

RUN groupadd --system buildcheck \
    && useradd --system --gid buildcheck --home-dir /opt/buildcheck-agent buildcheck \
    && chown -R buildcheck:buildcheck /opt/buildcheck-agent

USER buildcheck
WORKDIR /opt/buildcheck-agent/backend

EXPOSE 8000

CMD ["sh", "-c", "exec uvicorn competition_public_entry:app --host 0.0.0.0 --port ${PORT:-8000}"]
