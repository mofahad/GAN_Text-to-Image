FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt requirements-hosting.txt ./
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements-hosting.txt
COPY generate.py app.py ./
USER 1000
EXPOSE 7860
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--workers", "1", "--threads", "1", "--timeout", "120", "app:app"]
