# Salem Cat Identifier - Devbox Deployment Guide

## 🚀 Quick Deployment Steps

### 1. Create Deployment Package
```bash
# From your local machine in the find-salem directory
tar -czf salem-deploy.tar.gz simple_app.py simple_trainer.py models/ data/ README.md
```

### 2. Transfer to Devbox
Replace `YOUR_DEVBOX_ADDRESS` with your actual devbox address:
```bash
scp salem-deploy.tar.gz YOUR_DEVBOX_ADDRESS:~/
```

### 3. Setup on Devbox
SSH into your devbox and run:
```bash
ssh YOUR_DEVBOX_ADDRESS

# Extract files
tar -xzf salem-deploy.tar.gz
cd find-salem

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install streamlit pillow numpy scikit-learn matplotlib

# Run the app (accessible from anywhere)
streamlit run simple_app.py --server.address 0.0.0.0 --server.port 8501
```

### 4. Access Your App
Open in browser: `http://YOUR_DEVBOX_ADDRESS:8501`

## 🔧 Alternative: Docker Deployment

If you prefer Docker, here's a Dockerfile:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Copy requirements
COPY requirements.txt .
RUN pip install streamlit pillow numpy scikit-learn matplotlib

# Copy app files
COPY simple_app.py simple_trainer.py ./
COPY models/ ./models/
COPY data/ ./data/

EXPOSE 8501

CMD ["streamlit", "run", "simple_app.py", "--server.address", "0.0.0.0", "--server.port", "8501"]
```

Build and run:
```bash
docker build -t salem-identifier .
docker run -p 8501:8501 salem-identifier
```

## 📋 Files Needed for Deployment

- `simple_app.py` - Main Streamlit application
- `simple_trainer.py` - ML training module  
- `models/` - Directory with trained model
- `data/` - Dataset directories
- Dependencies: streamlit, pillow, numpy, scikit-learn, matplotlib

## 🌐 Network Configuration

Make sure your devbox allows:
- Inbound connections on port 8501
- Python 3.8+ installed
- Internet access for pip installs

## 🔍 Common Devbox Addresses to Try

- `devbox`
- `dev.local` 
- `YOUR_USERNAME-devbox`
- IP addresses in your local network (192.168.x.x or 10.0.x.x)

## ✅ Verification

Once deployed, you should see:
- Streamlit app running on port 8501
- All three tabs: Predict Salem, Model Info, Manage Data
- Model showing 93.2% test accuracy
- Ability to upload and predict on cat photos
