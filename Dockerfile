# Base image with Python 3.12
FROM python:3.12-slim

# Set working directory inside container
WORKDIR /app

# Copy your project files
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Default command (change train.py if needed)
CMD ["python", "-u", "train.py"]
