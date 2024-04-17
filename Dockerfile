FROM python:3.11

# Install necessary system packages
RUN apt-get update && \
    apt-get install -y libgl1-mesa-glx

# Install Python packages
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Streamlit app code into the image
COPY EndOfTermProject.py /app/

# Set the working directory
WORKDIR /app

# Run the Streamlit app
CMD ["streamlit", "run", "EndOfTermProject.py"]
