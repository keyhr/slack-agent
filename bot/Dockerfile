FROM python:3.12-slim

RUN pip install uv --no-cache-dir

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY bot.py agent.py tools.py config.py store.py news_digest.py ./

RUN mkdir -p /app/data

CMD ["uv", "run", "python", "bot.py"]
