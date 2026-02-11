# Image de base
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /code

# Dépendances: requirements.txt est à la racine
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copier le code API (ton main.py est dans API/)
COPY API/ ./API/

RUN mkdir -p /code/models /code/data

EXPOSE 8000

CMD ["uvicorn", "main:app", "--app-dir", "/code/API", "--host", "0.0.0.0", "--port", "8000", "--reload"]
