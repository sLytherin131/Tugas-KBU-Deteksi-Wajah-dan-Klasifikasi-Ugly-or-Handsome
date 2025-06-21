# Gunakan Python image dasar
FROM python:3.10-slim

# Install dependensi sistem (libGL untuk OpenCV)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set workdir
WORKDIR /app

# Salin semua file ke image
COPY . /app

# Install pip packages
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Jalankan aplikasi Flask dengan Gunicorn
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:8080"]
