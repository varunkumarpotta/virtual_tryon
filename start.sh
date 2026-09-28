#!/bin/bash
# ============================================
# AI Virtual Try-On — Start Script
# ============================================
# Starts both FastAPI backend and Vite frontend

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

echo -e "${PURPLE}"
echo "  ╔══════════════════════════════════════╗"
echo "  ║       AI Virtual Try-On              ║"
echo "  ║       Starting Application...        ║"
echo "  ╚══════════════════════════════════════╝"
echo -e "${NC}"

# Check .env
if [ ! -f ".env" ]; then
    echo -e "${BLUE}Creating .env from .env.example...${NC}"
    cp .env.example .env
fi

# Check Python venv
if [ ! -d "venv" ]; then
    echo -e "${RED}Python virtual environment not found.${NC}"
    echo "Run: python3.11 -m venv venv && source venv/bin/activate && pip install -r backend/requirements.txt"
    exit 1
fi

# Check node_modules
if [ ! -d "frontend/node_modules" ]; then
    echo -e "${BLUE}Installing frontend dependencies...${NC}"
    cd frontend && npm install && cd ..
fi

# Cleanup function
cleanup() {
    echo -e "\n${BLUE}Shutting down...${NC}"
    if [ ! -z "$BACKEND_PID" ]; then
        kill $BACKEND_PID 2>/dev/null || true
    fi
    if [ ! -z "$FRONTEND_PID" ]; then
        kill $FRONTEND_PID 2>/dev/null || true
    fi
    wait 2>/dev/null
    echo -e "${GREEN}Stopped.${NC}"
    exit 0
}

trap cleanup SIGINT SIGTERM

# Start Backend
echo -e "${GREEN}Starting FastAPI backend on http://localhost:8000 ...${NC}"
source venv/bin/activate
cd backend
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!
cd ..

# Wait for backend
echo -e "${BLUE}Waiting for backend to be ready...${NC}"
for i in {1..60}; do
    if curl -s http://localhost:8000/api/health > /dev/null 2>&1; then
        echo -e "${GREEN}Backend is ready!${NC}"
        break
    fi
    if [ $i -eq 60 ]; then
        echo -e "${RED}Backend failed to start. Check logs above.${NC}"
        cleanup
        exit 1
    fi
    sleep 2
done

# Start Frontend
echo -e "${GREEN}Starting Vite frontend on http://localhost:5173 ...${NC}"
cd frontend
npm run dev &
FRONTEND_PID=$!
cd ..

echo ""
echo -e "${PURPLE}════════════════════════════════════════${NC}"
echo -e "${GREEN}  ✓ Application is running!${NC}"
echo ""
echo -e "  Frontend: ${BLUE}http://localhost:5173${NC}"
echo -e "  Backend:  ${BLUE}http://localhost:8000${NC}"
echo -e "  Health:   ${BLUE}http://localhost:8000/api/health${NC}"
echo ""
echo -e "  Press ${RED}Ctrl+C${NC} to stop."
echo -e "${PURPLE}════════════════════════════════════════${NC}"
echo ""

# Wait for both
wait
