FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .
COPY predict.py .
COPY src ./src
COPY data/breast_cancer.csv ./data/breast_cancer.csv

CMD ["python", "main.py"]
