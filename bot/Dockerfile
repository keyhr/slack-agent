FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY bot.py agent.py tools.py config.py store.py ./

RUN mkdir -p /app/data

CMD ["uv", "run", "python", "bot.py"]
