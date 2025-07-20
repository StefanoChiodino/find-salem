#!/bin/bash

# Salem Cat Identifier - Deployment Script
# This script deploys the Salem identification system to a remote devbox
# Usage: ./deploy.sh [devbox-address] [remote-path]

set -e

echo "🚀 Deploying Salem Cat Identifier to devbox..."

# Configuration
PROJECT_NAME="find-salem"
USERNAME=$(whoami)

# Require devbox address as parameter
if [ $# -lt 1 ]; then
    echo "❌ Error: Devbox address is required"
    echo "Usage: ./deploy.sh <devbox-address> [remote-path]"
    echo "Example: ./deploy.sh dev27-uswest1adevc web"
    exit 1
fi

DEVBOX_HOST="$1"
echo "🎯 Using devbox address: $DEVBOX_HOST"

# Set remote path (can be overridden by second argument)
if [ $# -ge 2 ]; then
    REMOTE_PATH="$2"
else
    REMOTE_PATH="web"  # Default to web directory
fi

# Test SSH connection to devbox
echo "Testing connection to $DEVBOX_HOST..."
if ! ssh -o ConnectTimeout=5 -o BatchMode=yes "$DEVBOX_HOST" exit 2>/dev/null; then
    echo "❌ Cannot connect to $DEVBOX_HOST"
    echo "Please check:"
    echo "  - Devbox address is correct"
    echo "  - SSH key is configured"
    echo "  - Devbox is accessible"
    exit 1
fi
echo "✅ Connected to $DEVBOX_HOST"

# Create deployment package
echo "📦 Creating deployment package..."
DEPLOY_FILES=(
    "simple_app.py"
    "simple_trainer.py" 
    "requirements.txt"
    "README.md"
    "models/"
    "data/"
)

# Create temporary deployment directory
TEMP_DIR=$(mktemp -d)
echo "📁 Temporary directory: $TEMP_DIR"

# Copy files to temp directory
for file in "${DEPLOY_FILES[@]}"; do
    if [ -e "$file" ]; then
        echo "📋 Copying $file..."
        cp -r "$file" "$TEMP_DIR/"
    else
        echo "⚠️  Warning: $file not found, skipping..."
    fi
done

# Create a simple requirements file for deployment
cat > "$TEMP_DIR/deploy_requirements.txt" << EOF
streamlit>=1.28.0
pillow>=9.0.0
numpy>=1.21.0
scikit-learn>=1.3.0
matplotlib>=3.5.0
EOF

# Create deployment setup script
cat > "$TEMP_DIR/setup_remote.sh" << 'EOF'
#!/bin/bash
set -e

echo "🔧 Setting up Salem Cat Identifier on remote devbox..."

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r deploy_requirements.txt

echo "✅ Setup complete!"
echo "🚀 To run the app:"
echo "  source venv/bin/activate"
echo "  streamlit run simple_app.py --server.address 0.0.0.0 --server.port 8501"
EOF

chmod +x "$TEMP_DIR/setup_remote.sh"

# Transfer files to devbox
echo "📤 Transferring files to $DEVBOX_HOST:$REMOTE_PATH..."
ssh "$DEVBOX_HOST" "mkdir -p $REMOTE_PATH"
scp -r "$TEMP_DIR"/* "$DEVBOX_HOST:$REMOTE_PATH/"

# Run setup on remote
echo "🔧 Setting up environment on devbox..."
ssh "$DEVBOX_HOST" "cd $REMOTE_PATH && chmod +x setup_remote.sh && ./setup_remote.sh"

# Clean up
rm -rf "$TEMP_DIR"

echo "✅ Deployment complete!"
echo "🌐 To access your Salem identifier:"
echo "   ssh $DEVBOX_HOST"
echo "   cd $REMOTE_PATH"
echo "   source venv/bin/activate"
echo "   streamlit run simple_app.py --server.address 0.0.0.0 --server.port 8501"
echo ""
echo "🔗 Then open: http://$DEVBOX_HOST:8501"
