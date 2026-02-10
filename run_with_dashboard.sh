#!/bin/bash
# Run SimilanCao Trader with Dashboard
# Usage: ./run_with_dashboard.sh [--live]

echo "======================================"
echo "🐆 SimilanCao Trader with Dashboard"
echo "======================================"

# Activate virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
else
    source venv/bin/activate
fi

# Check if .env exists
if [ ! -f .env ]; then
    echo "⚠️  Warning: .env file not found. Copying from .env.example..."
    cp .env.example .env
    echo "Please edit .env with your API keys before running in live mode."
fi

# Start dashboard in background
echo "🌐 Starting dashboard on http://localhost:8080"
python dashboard.py &
DASHBOARD_PID=$!

# Wait a moment for dashboard to start
sleep 2

# Start bot with arguments
echo "🐆 Starting trading bot..."
if [ "$1" == "--live" ]; then
    echo "⚠️  LIVE TRADING MODE ENABLED"
    python bot.py --live
else
    echo "🧪 DRY-RUN MODE (paper trading)"
    python bot.py
fi

# Cleanup: kill dashboard when bot stops
echo ""
echo "Stopping dashboard..."
kill $DASHBOARD_PID 2>/dev/null

echo "👋 Shutdown complete"
