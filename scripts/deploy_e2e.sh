#!/usr/bin/env bash
# ==============================================================================
# SatQuery AI - E2E Networks Automated Production Deployment Script
# Tested on: Ubuntu 20.04 / 22.04 / 24.04 LTS (x86_64) on E2E Networks Cloud
# ==============================================================================
set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${CYAN}======================================================${NC}"
echo -e "${CYAN}   SatQuery AI - E2E Networks Deployment Engine       ${NC}"
echo -e "${CYAN}======================================================${NC}"

# 1. Verify sudo/root privileges
if [ "$EUID" -ne 0 ]; then
  echo -e "${YELLOW}[!] Escalating to sudo...${NC}"
  SUDO="sudo"
else
  SUDO=""
fi

# 2. Determine project directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

echo -e "${BLUE}[1/5] Checking and installing system prerequisites...${NC}"
$SUDO apt-get update -y
$SUDO apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    gnupg \
    lsb-release \
    git \
    ufw

# 3. Install Docker and Docker Compose if not present
if ! command -v docker &> /dev/null; then
    echo -e "${BLUE}[2/5] Docker not found. Installing Docker CE...${NC}"
    curl -fsSL https://get.docker.com | $SUDO sh
    $SUDO systemctl enable --now docker
    if [ -n "$SUDO" ] && [ -n "${SUDO_USER:-}" ]; then
        $SUDO usermod -aG docker "$SUDO_USER" || true
    fi
else
    echo -e "${GREEN}[2/5] Docker is already installed: $(docker --version)${NC}"
fi

# 4. Configure firewall ports (22 for SSH, 80 for HTTP, 8000 for SatQuery API)
echo -e "${BLUE}[3/5] Configuring firewall rules...${NC}"
if command -v ufw &> /dev/null; then
    $SUDO ufw allow 22/tcp comment 'SSH' || true
    $SUDO ufw allow 80/tcp comment 'HTTP' || true
    $SUDO ufw allow 8000/tcp comment 'SatQuery API' || true
    # Don't block active SSH session
    echo "y" | $SUDO ufw enable 2>/dev/null || true
fi

# 5. Build and launch SatQuery backend container
echo -e "${BLUE}[4/5] Building and starting SatQuery backend container...${NC}"
if docker compose version &> /dev/null; then
    COMPOSE_CMD="docker compose"
elif command -v docker-compose &> /dev/null; then
    COMPOSE_CMD="docker-compose"
else
    $SUDO apt-get install -y docker-compose-plugin
    COMPOSE_CMD="docker compose"
fi

$SUDO $COMPOSE_CMD -f docker-compose.e2e.yml down --remove-orphans || true
$SUDO $COMPOSE_CMD -f docker-compose.e2e.yml up -d --build

# 6. Health check polling
echo -e "${BLUE}[5/5] Polling health check on http://localhost:8000/health...${NC}"
MAX_RETRIES=20
RETRY_COUNT=0
HEALTHY=false

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/health || echo "000")
    if [ "$HTTP_CODE" = "200" ]; then
        HEALTHY=true
        break
    fi
    echo -n "."
    sleep 3
    RETRY_COUNT=$((RETRY_COUNT + 1))
done
echo ""

PUBLIC_IP=$(curl -s --max-time 4 https://api.ipify.org || curl -s --max-time 4 ifconfig.me || hostname -I | awk '{print $1}')

if [ "$HEALTHY" = true ]; then
    echo -e "${GREEN}======================================================${NC}"
    echo -e "${GREEN}   Deployment Succeeded! SatQuery Backend is Online!  ${NC}"
    echo -e "${GREEN}======================================================${NC}"
    echo -e "Public API URL:      ${CYAN}http://${PUBLIC_IP}:8000${NC}"
    echo -e "Health Check:        ${CYAN}http://${PUBLIC_IP}:8000/health${NC}"
    echo -e "Interactive Swagger: ${CYAN}http://${PUBLIC_IP}:8000/docs${NC}"
    echo ""
    echo -e "${YELLOW}Next Steps for Frontend Configuration:${NC}"
    echo -e "In your ${CYAN}frontend/.env${NC} or Vercel Environment Variables, set:"
    echo -e "--------------------------------------------------------"
    echo -e "${GREEN}VITE_API_URL=http://${PUBLIC_IP}:8000${NC}"
    echo -e "--------------------------------------------------------"
else
    echo -e "${RED}Backend container started but health check timed out.${NC}"
    echo -e "${YELLOW}Displaying recent container logs:${NC}"
    $SUDO $COMPOSE_CMD -f docker-compose.e2e.yml logs --tail=40
    exit 1
fi
