# BatteryForgeAI Deployment Guide

## Overview
BatteryForgeAI is deployed as a full-stack application with:
- **Frontend**: React/Vite deployed on Vercel
- **Backend**: FastAPI deployed on Heroku

## Current Status
✅ Frontend deployed: https://frontend-wheat-xi-78.vercel.app
🔄 Backend ready for deployment

## Backend Deployment (Heroku)

### Prerequisites
1. Heroku CLI installed: https://devcenter.heroku.com/articles/heroku-cli
2. Heroku account
3. GitHub repository connected

### Environment Variables Required
Set these in Heroku dashboard or via CLI:
```
GEMINI_API_KEY=your_gemini_api_key
RUNPOD_ENDPOINT=your_runpod_endpoint
RUNPOD_API_KEY=your_runpod_api_key
```

### Deployment Steps

#### Option 1: Using the Deploy Script (Recommended)
```bash
cd backend
chmod +x deploy.sh
./deploy.sh
```

#### Option 2: Manual Deployment
```bash
# Login to Heroku
heroku login

# Create app
heroku create batteryforgeai --stack=container

# Set environment variables
heroku config:set GEMINI_API_KEY="your_key_here"
heroku config:set RUNPOD_ENDPOINT="your_endpoint_here"
heroku config:set RUNPOD_API_KEY="your_key_here"

# Deploy
git push heroku main
```

### Alternative Platforms
If Heroku doesn't work, try:
- **Railway**: `railway login && railway link && railway up`
- **Render**: Connect GitHub repo, set environment variables
- **Fly.io**: `fly launch` in backend directory

## Frontend Configuration

### Update API URL
In Vercel dashboard, set environment variable:
```
VITE_API_BASE_URL=https://your-backend-url.herokuapp.com
```

### Rebuild Frontend
```bash
cd frontend
npm run build
# Vercel will auto-deploy on git push
```

## Testing Deployment

### Backend Health Check
```bash
curl https://your-backend-url.herokuapp.com/health
```

### Frontend-Backend Integration
1. Open frontend URL
2. Check browser console for API errors
3. Test core features (PDF upload, fleet monitoring)

## Troubleshooting

### Common Issues
1. **CORS errors**: Check backend CORS settings in main.py
2. **Environment variables**: Verify all required vars are set
3. **Port binding**: Backend must bind to $PORT on Heroku
4. **Dependencies**: Check requirements.txt is complete

### Logs
```bash
# Heroku logs
heroku logs --tail -a your-app-name

# Vercel logs
vercel logs your-frontend-url
```

## Production URLs
- Frontend: https://frontend-wheat-xi-78.vercel.app
- Backend: https://[your-heroku-app].herokuapp.com
- API Docs: https://[your-heroku-app].herokuapp.com/docs