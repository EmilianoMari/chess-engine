"""Gunicorn configuration for LC0 Service."""

import os

# Server socket
bind = f"{os.getenv('HOST', '0.0.0.0')}:{os.getenv('PORT', '8001')}"
backlog = 64

# Worker processes
workers = 1  # Single worker because LC0 is stateful
worker_class = "uvicorn.workers.UvicornWorker"
timeout = 300  # Long timeout for analysis
graceful_timeout = 30

# Logging
accesslog = "-"
errorlog = "-"
loglevel = "info"

# Process naming
proc_name = "lc0-service"
