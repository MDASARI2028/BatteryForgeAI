#!/bin/bash

echo "🚀 Deploying BatteryForgeAI to Production"
echo "=========================================="

# Check if Heroku CLI is installed
if ! command -v heroku &> /dev/null; then
    echo "❌ Heroku CLI not found. Install from: https://devcenter.heroku.com/articles/heroku-cli"
    exit 1
fi

# Check if logged in
if ! heroku auth:whoami &> /dev/null; then
    echo "❌ Not logged in to Heroku. Run: heroku login"
    exit 1
fi

echo "✅ Heroku CLI ready"

# Create Heroku app
echo "📦 Creating Heroku app..."
HEROKU_APP_NAME="${HEROKU_APP_NAME:-batteryforgeai}"
heroku create $HEROKU_APP_NAME --stack=container

# Set environment variables
echo "🔧 Setting environment variables..."
heroku config:set GEMINI_API_KEY="$GEMINI_API_KEY" -a $HEROKU_APP_NAME
heroku config:set RUNPOD_ENDPOINT="$RUNPOD_ENDPOINT" -a $HEROKU_APP_NAME
heroku config:set RUNPOD_API_KEY="$RUNPOD_API_KEY" -a $HEROKU_APP_NAME

# Deploy
echo "🚀 Deploying..."
git push heroku main

# Open app
echo "🌐 Opening app..."
heroku open -a $HEROKU_APP_NAME

echo "✅ Deployment complete!"
echo "App URL: https://$HEROKU_APP_NAME.herokuapp.com"
echo "API Docs: https://$HEROKU_APP_NAME.herokuapp.com/docs"