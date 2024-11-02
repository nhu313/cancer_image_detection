# Use the official Python image from DockerHub
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the necessary packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    pip install --no-cache-dir -r requirements.txt && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy the entire project into the container
COPY . /app

# Expose port 5001 for Flask
EXPOSE 5001

# Set environment variables (optional)
ENV FLASK_APP=app/main.py
ENV FLASK_ENV=development

# Command to run the Flask app
CMD ["flask", "run", "--host=0.0.0.0"]
