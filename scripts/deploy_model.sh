#!/bin/bash

# Script to deploy a trained ML model to Cloud Run

set -e

# Check if required arguments are provided
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <project_id> <region> [model_path]"
    echo "Example: $0 my-gcp-project us-central1 ./models/serving_model"
    exit 1
fi

# Parse arguments
PROJECT_ID=$1
REGION=$2
MODEL_PATH=${3:-"./models/serving_model"}

# Set variables
IMAGE_NAME="ml-model-server"
IMAGE_TAG="latest"
SERVICE_NAME="ml-prediction-service"
ARTIFACT_REGISTRY="${REGION}-docker.pkg.dev/${PROJECT_ID}/ml-models"

echo "Deploying model from ${MODEL_PATH} to ${PROJECT_ID} in ${REGION}"

# Ensure Docker is installed
if ! command -v docker &> /dev/null; then
    echo "Docker is required but not installed. Please install Docker first."
    exit 1
fi

# Ensure gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo "gcloud is required but not installed. Please install Google Cloud SDK first."
    exit 1
fi

# Ensure the user is authenticated with gcloud
echo "Checking gcloud authentication..."
gcloud auth print-access-token &> /dev/null || {
    echo "Not authenticated with gcloud. Please run 'gcloud auth login' first."
    exit 1
}

# Set the GCP project
echo "Setting GCP project to ${PROJECT_ID}..."
gcloud config set project ${PROJECT_ID}

# Create Artifact Registry repository if it doesn't exist
echo "Creating Artifact Registry repository if it doesn't exist..."
gcloud artifacts repositories create ml-models \
    --repository-format=docker \
    --location=${REGION} \
    --description="ML model container images" \
    || true

# Build the Docker image
echo "Building Docker image..."
docker build -t ${IMAGE_NAME}:${IMAGE_TAG} \
    --build-arg MODEL_PATH=${MODEL_PATH} \
    -f Dockerfile .

# Tag the image for Artifact Registry
echo "Tagging image for Artifact Registry..."
docker tag ${IMAGE_NAME}:${IMAGE_TAG} ${ARTIFACT_REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}

# Configure Docker for Artifact Registry
echo "Configuring Docker for Artifact Registry..."
gcloud auth configure-docker ${REGION}-docker.pkg.dev --quiet

# Push the image to Artifact Registry
echo "Pushing image to Artifact Registry..."
docker push ${ARTIFACT_REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}

# Deploy to Cloud Run
echo "Deploying to Cloud Run..."
gcloud run deploy ${SERVICE_NAME} \
    --image=${ARTIFACT_REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG} \
    --platform=managed \
    --region=${REGION} \
    --allow-unauthenticated \
    --memory=2Gi \
    --cpu=1 \
    --min-instances=1 \
    --max-instances=5 \
    --port=8080

# Get the service URL
SERVICE_URL=$(gcloud run services describe ${SERVICE_NAME} \
    --platform=managed \
    --region=${REGION} \
    --format='value(status.url)')

echo "\nDeployment completed successfully!"
echo "Service URL: ${SERVICE_URL}"
echo "\nTest the service with:"
echo "curl -X POST ${SERVICE_URL}/predict \
    -H \"Content-Type: application/json\" \
    -d '{"quantity": 5, "unit_price": 50.0, "discount": 0.1, "customer_age": 35, "customer_gender": "M", "store_id": "STORE_01", "payment_method": "Credit Card", "transaction_hour": 14}'"
