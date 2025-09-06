# 🚀 Deployment Checklist for CogniBank API

## ✅ Pre-Deployment Checklist

### 1. Environment Setup
- [ ] OpenAI API key is valid and has credits
- [ ] MongoDB Atlas cluster is running
- [ ] MongoDB user has proper permissions
- [ ] Network access is configured (0.0.0.0/0 for testing)

### 2. Code Cleanup ✅ COMPLETED
- [x] Removed test files
- [x] Removed development scripts
- [x] Removed unused documentation
- [x] Updated .dockerignore
- [x] Cleaned requirements.txt

### 3. Configuration Files ✅ READY
- [x] Dockerfile optimized
- [x] fly.toml configured
- [x] requirements.txt minimal
- [x] .env.production template

## 🚀 Deployment Commands

### 1. Install Fly.io CLI
```bash
# Windows
iwr https://fly.io/install.ps1 -useb | iex

# Verify installation
flyctl version
```

### 2. Login and Create App
```bash
# Login to Fly.io
flyctl auth login

# Create new app (or use existing)
flyctl apps create cognibank-api
```

### 3. Set Environment Variables
```bash
# Set OpenAI API key
flyctl secrets set OPENAI_API_KEY="sk-proj-your-key-here" --app cognibank-api

# Set MongoDB URI
flyctl secrets set MONGODB_URI="mongodb+srv://nandankhawale_db_user:0P154bJbSm87GmZm@cognibankdata.0ghzgfa.mongodb.net/loan_applications?retryWrites=true&w=majority&appName=CogniBankData" --app cognibank-api

# Set database name
flyctl secrets set MONGODB_DATABASE="loan_applications" --app cognibank-api
```

### 4. Deploy Application
```bash
# Deploy to Fly.io
flyctl deploy --app cognibank-api
```

### 5. Verify Deployment
```bash
# Check status
flyctl status --app cognibank-api

# View logs
flyctl logs --app cognibank-api

# Test health endpoint
curl https://cognibank-api.fly.dev/health
```

## 🔍 Post-Deployment Testing

### API Endpoints to Test:
1. **Health Check**: `GET https://cognibank-api.fly.dev/health`
2. **Loan Types**: `GET https://cognibank-api.fly.dev/loan-types`
3. **Start Chat**: `POST https://cognibank-api.fly.dev/chat/start`
4. **Admin Stats**: `GET https://cognibank-api.fly.dev/admin/stats`

### Test Commands:
```bash
# Health check
curl https://cognibank-api.fly.dev/health

# Get loan types
curl https://cognibank-api.fly.dev/loan-types

# Start education loan chat
curl -X POST https://cognibank-api.fly.dev/chat/start \
  -H "Content-Type: application/json" \
  -d '{"loan_type": "education"}'
```

## 📊 Monitoring

### View Application Metrics:
```bash
# Real-time logs
flyctl logs --app cognibank-api -f

# Application metrics
flyctl metrics --app cognibank-api

# Machine status
flyctl machine list --app cognibank-api
```

## 🔧 Troubleshooting

### Common Issues:

1. **MongoDB Connection Failed**
   - Check MongoDB Atlas IP whitelist
   - Verify connection string format
   - Test connection locally first

2. **OpenAI API Errors**
   - Verify API key is valid
   - Check account has credits
   - Test API key locally

3. **Application Won't Start**
   - Check logs: `flyctl logs --app cognibank-api`
   - Verify environment variables: `flyctl secrets list --app cognibank-api`
   - Check Dockerfile syntax

4. **High Memory Usage**
   - Scale memory: `flyctl scale memory 2048 --app cognibank-api`
   - Monitor usage: `flyctl metrics --app cognibank-api`

## 🎯 Success Criteria

- [ ] Health endpoint returns 200 OK
- [ ] All 6 loan types are available
- [ ] Chat sessions can be started
- [ ] MongoDB connection is working
- [ ] Admin endpoints return data
- [ ] Application logs show no errors
- [ ] Response times are under 2 seconds

## 📱 Frontend Integration

Once deployed, update your frontend to use:
```javascript
const API_BASE_URL = 'https://cognibank-api.fly.dev';
```

## 🔄 Updates and Maintenance

### Deploy Updates:
```bash
flyctl deploy --app cognibank-api
```

### Scale Application:
```bash
# Scale instances
flyctl scale count 2 --app cognibank-api

# Scale memory
flyctl scale memory 2048 --app cognibank-api
```

### Backup and Recovery:
- MongoDB Atlas handles automatic backups
- Application is stateless - easy to redeploy
- Environment variables are stored in Fly.io secrets