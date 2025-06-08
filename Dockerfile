# Use official RDKit base image with Python 3.10
FROM rdkit/rdkit:2023.03.2

WORKDIR /app

# Copy app code and requirements
COPY . /app

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Install system dependencies (for matplotlib, rendering)
RUN apt-get update && apt-get install -y libgl1-mesa-glx && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose Streamlit port
EXPOSE 8501

# Run Streamlit app
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
