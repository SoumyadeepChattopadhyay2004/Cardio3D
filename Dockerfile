# Not built or run by the author of this change: treat as a starting point and report problems.
FROM python:3.12-slim
WORKDIR /app
COPY requirements.lock requirements.txt ./
RUN pip install --no-cache-dir -r requirements.lock
COPY cardio ./cardio
COPY static ./static
COPY artifacts ./artifacts
EXPOSE 8000
CMD ["uvicorn", "cardio.serve:app", "--host", "0.0.0.0", "--port", "8000"]
