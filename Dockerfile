FROM python:3.12-slim
RUN pip install --no-cache-dir duckdb numpy scipy pandas
WORKDIR /app
COPY main.py /app/main.py
ENV PYTHONUNBUFFERED=1
CMD ["python", "/app/main.py"]
