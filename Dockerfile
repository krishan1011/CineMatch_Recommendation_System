FROM python:3.11-slim

WORKDIR /code
COPY requirements-deploy.txt .
RUN pip install --no-cache-dir -r requirements-deploy.txt
COPY . .

EXPOSE 7860
CMD ["gunicorn", "app.app:app", "--bind", "0.0.0.0:7860", "--workers", "1", "--timeout", "120"]
