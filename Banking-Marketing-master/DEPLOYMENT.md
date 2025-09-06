# Deployment Guide

This guide covers deploying the Cognibank frontend application in various environments.

## Prerequisites

- Backend API running and accessible
- Node.js 18+ (for local deployment)
- Docker (for containerized deployment)

## Environment Variables

Set these environment variables based on your deployment:

```bash
# Required
NEXT_PUBLIC_API_URL=http://your-backend-url:8001

# Optional
NODE_ENV=production
NEXT_PUBLIC_APP_NAME=Cognibank
NEXT_PUBLIC_APP_VERSION=2.0.0
```

## Local Development

1. **Install dependencies**:
   ```bash
   npm install
   ```

2. **Set environment variables**:
   ```bash
   cp .env.local.example .env.local
   # Edit .env.local with your backend URL
   ```

3. **Start development server**:
   ```bash
   npm run dev
   ```

4. **Access application**:
   Open `http://localhost:3000`

## Production Deployment

### Option 1: Node.js Deployment

1. **Build the application**:
   ```bash
   npm run build
   ```

2. **Start production server**:
   ```bash
   npm run start
   ```

3. **Use PM2 for process management** (recommended):
   ```bash
   npm install -g pm2
   pm2 start npm --name "cognibank-frontend" -- start
   pm2 save
   pm2 startup
   ```

### Option 2: Docker Deployment

1. **Build Docker image**:
   ```bash
   docker build -t cognibank-frontend .
   ```

2. **Run container**:
   ```bash
   docker run -d \
     --name cognibank-frontend \
     -p 3000:3000 \
     -e NEXT_PUBLIC_API_URL=http://your-backend:8001 \
     cognibank-frontend
   ```

### Option 3: Docker Compose

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  frontend:
    build: .
    ports:
      - "3000:3000"
    environment:
      - NEXT_PUBLIC_API_URL=http://backend:8001
    depends_on:
      - backend
    restart: unless-stopped

  backend:
    image: your-backend-image
    ports:
      - "8001:8001"
    environment:
      - PORT=8001
    restart: unless-stopped
```

Run with:
```bash
docker-compose up -d
```

## Cloud Deployment

### Vercel (Recommended for Next.js)

1. **Connect repository to Vercel**
2. **Set environment variables in Vercel dashboard**:
   - `NEXT_PUBLIC_API_URL`
3. **Deploy automatically on push**

### Netlify

1. **Build command**: `npm run build`
2. **Publish directory**: `.next`
3. **Set environment variables**:
   - `NEXT_PUBLIC_API_URL`

### AWS EC2

1. **Launch EC2 instance** (Ubuntu 20.04+)
2. **Install Node.js**:
   ```bash
   curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
   sudo apt-get install -y nodejs
   ```

3. **Clone and setup application**:
   ```bash
   git clone <your-repo>
   cd Banking-Marketing-master
   npm install
   npm run build
   ```

4. **Setup PM2**:
   ```bash
   sudo npm install -g pm2
   pm2 start npm --name "cognibank-frontend" -- start
   pm2 startup
   pm2 save
   ```

5. **Setup Nginx reverse proxy**:
   ```nginx
   server {
       listen 80;
       server_name your-domain.com;

       location / {
           proxy_pass http://localhost:3000;
           proxy_http_version 1.1;
           proxy_set_header Upgrade $http_upgrade;
           proxy_set_header Connection 'upgrade';
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
           proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
           proxy_set_header X-Forwarded-Proto $scheme;
           proxy_cache_bypass $http_upgrade;
       }
   }
   ```

### Google Cloud Platform

1. **Create App Engine app.yaml**:
   ```yaml
   runtime: nodejs18
   
   env_variables:
     NEXT_PUBLIC_API_URL: "https://your-backend-url"
   
   automatic_scaling:
     min_instances: 1
     max_instances: 10
   ```

2. **Deploy**:
   ```bash
   gcloud app deploy
   ```

## SSL/HTTPS Setup

### Using Certbot (Let's Encrypt)

```bash
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com
```

### Using Cloudflare

1. Add domain to Cloudflare
2. Update DNS to point to your server
3. Enable SSL/TLS encryption

## Performance Optimization

### 1. Enable Compression

In Nginx:
```nginx
gzip on;
gzip_vary on;
gzip_min_length 1024;
gzip_types text/plain text/css text/xml text/javascript application/javascript application/xml+rss application/json;
```

### 2. Caching Headers

```nginx
location /_next/static/ {
    expires 1y;
    add_header Cache-Control "public, immutable";
}

location /static/ {
    expires 1y;
    add_header Cache-Control "public, immutable";
}
```

### 3. CDN Setup

Use Cloudflare, AWS CloudFront, or similar CDN for static assets.

## Monitoring

### Health Check Endpoint

The application runs on port 3000. Health check:
```bash
curl http://localhost:3000
```

### PM2 Monitoring

```bash
pm2 status
pm2 logs cognibank-frontend
pm2 monit
```

### Docker Health Check

Add to Dockerfile:
```dockerfile
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:3000 || exit 1
```

## Troubleshooting

### Common Issues

1. **Build Failures**
   - Check Node.js version (18+)
   - Clear node_modules and reinstall
   - Check for syntax errors

2. **API Connection Issues**
   - Verify NEXT_PUBLIC_API_URL is correct
   - Check CORS settings on backend
   - Ensure backend is accessible from frontend

3. **Performance Issues**
   - Enable compression
   - Use CDN for static assets
   - Optimize images

### Logs

- **Development**: Check browser console
- **Production**: Check PM2 logs or Docker logs
- **Server**: Check Nginx error logs

```bash
# PM2 logs
pm2 logs cognibank-frontend

# Docker logs
docker logs cognibank-frontend

# Nginx logs
sudo tail -f /var/log/nginx/error.log
```

## Security Considerations

1. **Environment Variables**: Never commit sensitive data
2. **HTTPS**: Always use HTTPS in production
3. **CORS**: Configure backend CORS properly
4. **Headers**: Set security headers in Nginx/reverse proxy

Example security headers:
```nginx
add_header X-Frame-Options "SAMEORIGIN" always;
add_header X-Content-Type-Options "nosniff" always;
add_header X-XSS-Protection "1; mode=block" always;
add_header Referrer-Policy "strict-origin-when-cross-origin" always;
```

## Backup and Recovery

1. **Code**: Use Git for version control
2. **Environment**: Document all environment variables
3. **Database**: Backend handles data backup
4. **Static Assets**: Backup public folder if customized

## Scaling

### Horizontal Scaling

Use load balancer with multiple instances:

```yaml
# docker-compose.yml
version: '3.8'

services:
  frontend-1:
    build: .
    environment:
      - NEXT_PUBLIC_API_URL=http://backend:8001

  frontend-2:
    build: .
    environment:
      - NEXT_PUBLIC_API_URL=http://backend:8001

  nginx:
    image: nginx
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - frontend-1
      - frontend-2
```

### Vertical Scaling

Increase server resources (CPU, RAM) as needed.

## Support

For deployment issues:
1. Check logs first
2. Verify environment variables
3. Test backend connectivity
4. Check firewall/security groups
5. Review this deployment guide