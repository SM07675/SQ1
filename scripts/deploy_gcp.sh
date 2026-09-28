#!/usr/bin/env bash
# ==============================================================================
# SatQuery AI - Google Cloud Run 1-Click Deployment Script
# Run this inside Google Cloud Shell (console.cloud.google.com -> [ >_ ])
# ==============================================================================
set -euo pipefail

CYAN='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${CYAN}======================================================${NC}"
echo -e "${CYAN}   SatQuery AI -> Google Cloud Run Deployment Engine  ${NC}"
echo -e "${CYAN}======================================================${NC}"

# 1. Check gcloud CLI
if ! command -v gcloud &> /dev/null; then
    echo -e "${RED}[!] gcloud CLI is not installed.${NC}"
    echo -e "Please run this script inside Google Cloud Shell in your browser:"
    echo -e "https://console.cloud.google.com/ (Click the [ >_ ] terminal icon top-right)"
    exit 1
fi

PROJECT_ID=$(gcloud config get-value project 2>/dev/null || echo "")
if [ -z "$PROJECT_ID" ] || [ "$PROJECT_ID" = "(unset)" ]; then
    echo -e "${YELLOW}[?] No default GCP project selected.${NC}"
    gcloud projects list
    read -p "Enter your Google Cloud Project ID: " PROJECT_ID
    gcloud config set project "$PROJECT_ID"
fi

echo -e "${GREEN}Active Project: ${PROJECT_ID}${NC}"

# 2. Select Region
REGION="asia-south1" # Mumbai, India (ultra low latency)
echo -e "${BLUE}Target Region: ${REGION} (Mumbai, India)${NC}"

# 3. Enable necessary Google Cloud APIs
echo -e "${BLUE}[1/3] Enabling required Google Cloud APIs...${NC}"
gcloud services enable \
    run.googleapis.com \
    artifactregistry.googleapis.com \
    cloudbuild.googleapis.com

# 4. Deploy to Cloud Run with Scale-to-Zero
echo -e "${BLUE}[2/3] Building and deploying SatQuery AI container to Cloud Run...${NC}"
echo -e "${YELLOW}(Configured with --min-instances 0; build, storage, and network charges may still apply.)${NC}"

gcloud run deploy satquery-backend \
    --source . \
    --platform managed \
    --region "$REGION" \
    --port 8080 \
    --memory 16Gi \
    --cpu 8 \
    --timeout 300 \
    --concurrency 80 \
    --min-instances 0 \
    --max-instances 2 \
    --allow-unauthenticated \
    --set-env-vars SATQUERY_ENV=production,SATQUERY_DEVICE=cpu

# 5. Retrieve Service URL
SERVICE_URL=$(gcloud run services describe satquery-backend --platform managed --region "$REGION" --format 'value(status.url)')

echo -e "${GREEN}======================================================${NC}"
echo -e "${GREEN}   Deployment Succeeded! SatQuery Backend is Online!  ${NC}"
echo -e "${GREEN}======================================================${NC}"
echo -e "Service URL:         ${CYAN}${SERVICE_URL}${NC}"
echo -e "Health Check:        ${CYAN}${SERVICE_URL}/health${NC}"
echo -e "Interactive Swagger: ${CYAN}${SERVICE_URL}/docs${NC}"
echo ""
echo -e "${YELLOW}Next Steps to Connect Frontend:${NC}"
echo -e "1. Update frontend/.env.production:"
echo -e "   ${GREEN}VITE_API_URL=${SERVICE_URL}${NC}"
echo -e "2. Run: powershell .\\scripts\\deploy_vercel.ps1"
echo ""
