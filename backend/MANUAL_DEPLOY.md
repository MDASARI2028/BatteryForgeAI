# Manual Backend Deployment Guide

Since Heroku CLI is not installed, follow these manual steps:

## Step 1: Install Heroku CLI
1. Download from: https://devcenter.heroku.com/articles/heroku-cli
2. Install and restart terminal
3. Run: `heroku login`

## Step 2: Create Heroku App
```bash
cd backend
heroku create batteryforgeai --stack=container
```

## Step 3: Set Environment Variables
```bash
heroku config:set GEMINI_API_KEY="your_actual_gemini_key"
heroku config:set RUNPOD_ENDPOINT="your_actual_runpod_endpoint"
heroku config:set RUNPOD_API_KEY="your_actual_runpod_key"
```

## Step 4: Deploy
```bash
git add .
git commit -m "Deploy backend to Heroku"
git push heroku main
```

## Step 5: Get App URL
```bash
heroku info -a batteryforgeai
```

## Step 6: Update Frontend
In Vercel dashboard, set:
```
VITE_API_BASE_URL=https://batteryforgeai.herokuapp.com
```

## Alternative: Use Railway (Easier)
1. Go to https://railway.app
2. Connect GitHub repo
3. Set environment variables in Railway dashboard
4. Deploy

Railway is often easier than Heroku for containerized apps.