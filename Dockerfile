FROM python:3.14-slim-bookworm
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    CAROTID_DICOM_ROOT=/data/dicom CAROTID_PROJECTS_ROOT=/data/projects
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libgomp1 libxrender1 libsm6 libxext6 && rm -rf /var/lib/apt/lists/*
COPY server/requirements*.txt server/constraints.txt /app/server/
RUN pip install --no-cache-dir -r server/requirements-dev.txt
COPY server /app/server
COPY carotid_seg /app/carotid_seg
COPY tests /app/tests
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
