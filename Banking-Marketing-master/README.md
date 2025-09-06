# Cognibank Frontend

A modern Next.js banking application with loan management and admin dashboard capabilities.

## Features

- **Loan Applications**: Interactive chatbot for loan applications
- **Admin Dashboard**: Comprehensive admin panel with statistics, applications, and reports
- **Loan Calculator**: Real-time EMI calculator
- **Responsive Design**: Mobile-first responsive design
- **Real-time Data**: Live integration with FastAPI backend

## Tech Stack

- **Framework**: Next.js 15.5.2
- **React**: 19.1.0
- **Styling**: Tailwind CSS 4
- **HTTP Client**: Axios
- **Language**: JavaScript/JSX

## Prerequisites

- Node.js 18+ 
- npm or yarn
- Backend API running on port 8001 (or configured port)

## Installation

1. **Clone and navigate to frontend directory**:
   ```bash
   cd Banking-Marketing-master
   ```

2. **Install dependencies**:
   ```bash
   npm install
   ```

3. **Set up environment variables**:
   ```bash
   cp .env.local.example .env.local
   ```
   
   Update `.env.local` with your backend API URL:
   ```
   NEXT_PUBLIC_API_URL=http://localhost:8001
   ```

4. **Start development server**:
   ```bash
   npm run dev
   ```

5. **Open browser**:
   Navigate to `http://localhost:3000`

## Available Scripts

- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run start` - Start production server
- `npm run lint` - Run ESLint

## Project Structure

```
Banking-Marketing-master/
├── app/                    # Next.js app directory
│   ├── admin/             # Admin dashboard pages
│   ├── loans/             # Loan pages
│   ├── layout.js          # Root layout
│   └── page.js            # Home page
├── components/            # Reusable components
│   ├── admin/            # Admin-specific components
│   ├── Chatbot.jsx       # Loan application chatbot
│   ├── Header.jsx        # Site header
│   └── Footer.jsx        # Site footer
├── config/               # Configuration files
│   └── api.js           # API configuration
├── public/              # Static assets
└── styles/              # Global styles
```

## API Integration

The frontend integrates with the FastAPI backend through:

### Loan Services
- `/loan-types` - Get available loan types
- `/chat/start` - Start loan application session
- `/chat/message` - Send messages in loan application

### Admin Services
- `/admin/stats` - Get loan statistics
- `/admin/applications/{loan_type}` - Get applications by type
- `/admin/exports` - Get export information
- `/admin/export/{loan_type}` - Download CSV reports
- `/admin/generate-report/{loan_type}` - Generate new reports

## Key Components

### Chatbot (`/components/Chatbot.jsx`)
- Interactive loan application interface
- Real-time communication with backend
- Session management
- Loan type selection and application flow

### Admin Dashboard (`/app/admin/`)
- Statistics overview for all loan types
- Application management and viewing
- CSV report generation and download
- Real-time data updates

### Loan Calculator (`/app/loans/`)
- Interactive EMI calculator
- Dynamic loan type loading from backend
- Real-time calculations

## Environment Configuration

### Development (`.env.local`)
```
NEXT_PUBLIC_API_URL=http://localhost:8001
NODE_ENV=development
```

### Production (`.env.production`)
```
NEXT_PUBLIC_API_URL=https://your-backend-domain.com
NODE_ENV=production
```

## Deployment

### Development
```bash
npm run dev
```

### Production Build
```bash
npm run build
npm run start
```

### Docker Deployment
```bash
# Build image
docker build -t cognibank-frontend .

# Run container
docker run -p 3000:3000 -e NEXT_PUBLIC_API_URL=http://your-backend:8001 cognibank-frontend
```

## Backend Compatibility

This frontend is designed to work with the FastAPI backend that includes:

- **Loan Types**: education, home, personal, business, gold, car
- **Admin Endpoints**: Statistics, applications, exports
- **Chat System**: Session-based loan applications
- **Data Storage**: MongoDB or local file storage

## Troubleshooting

### Common Issues

1. **API Connection Failed**
   - Ensure backend is running on configured port
   - Check CORS settings in backend
   - Verify API URL in environment variables

2. **Chatbot Not Working**
   - Check backend `/health` endpoint
   - Verify session management in backend
   - Check browser console for errors

3. **Admin Dashboard Empty**
   - Ensure backend has loan application data
   - Check admin endpoints are accessible
   - Verify authentication if implemented

### Development Tips

- Use browser dev tools to monitor API calls
- Check backend logs for API errors
- Use React DevTools for component debugging

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## License

This project is licensed under the MIT License.