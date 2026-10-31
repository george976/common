// index.js
const express = require('express');
const cors = require('cors');

const app = express();
const PORT = process.env.PORT || 3000;

// ============================================
// Middleware
// ============================================

// Enable CORS for all routes
app.use(cors({
  origin: '*', // Change to specific origin(s) in production, e.g. ['https://yourdomain.com']
  methods: ['GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'OPTIONS'],
  allowedHeaders: ['Content-Type', 'Authorization'],
}));

// Parse JSON request bodies
app.use(express.json());

// Parse URL-encoded request bodies
app.use(express.urlencoded({ extended: true }));

// Simple request logger
app.use((req, res, next) => {
  console.log(`[${new Date().toISOString()}] ${req.method} ${req.url}`);
  next();
});

// ============================================
// Routes
// ============================================

// Health check / root API
app.get('/', (req, res) => {
  res.json({
    success: true,
    message: 'hi..',
  });
});

// Example: /api/hi
app.get('/api/hi', (req, res) => {
  res.status(200).json({
    success: true,
    message: 'hi..',
    timestamp: new Date().toISOString(),
  });
});

// Example POST route
app.post('/api/echo', (req, res) => {
  res.status(200).json({
    success: true,
    received: req.body,
  });
});

// ============================================
// 404 Handler
// ============================================
app.use((req, res) => {
  res.status(404).json({
    success: false,
    message: 'Route not found',
  });
});

// ============================================
// Global Error Handler
// ============================================
app.use((err, req, res, next) => {
  console.error('Error:', err.stack);
  res.status(err.status || 500).json({
    success: false,
    message: err.message || 'Internal Server Error',
  });
});

// ============================================
// Start Server
// ============================================
app.listen(PORT, () => {
  console.log(`🚀 Server running on http://localhost:${PORT}`);
});

module.exports = app;