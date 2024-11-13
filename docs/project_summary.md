# Enterprise ML Pipeline on GCP - Project Summary

## Overview

This project implements a production-quality Enterprise Machine Learning Pipeline on Google Cloud Platform (GCP). The pipeline automates the entire ML workflow from data ingestion and validation to model training, evaluation, and deployment. It's designed to showcase professional ML engineering practices for job applications.

## Key Components

### 1. Data Processing

- **BigQuery Integration**: Efficient data storage and querying with optimized pipelines
- **Data Validation**: Automated validation using TensorFlow Data Validation (TFDV)
- **Feature Engineering**: Consistent preprocessing with TensorFlow Transform

### 2. ML Pipeline

- **TFX Pipeline**: End-to-end ML workflow using TensorFlow Extended
- **Pipeline Components**: ExampleGen, StatisticsGen, SchemaGen, ExampleValidator, Transform, Trainer, Evaluator, and Pusher
- **Local & Cloud Execution**: Support for both local execution and Vertex AI pipeline runs

### 3. Model Training & Evaluation

- **TensorFlow Model**: Neural network with support for numeric and categorical features
- **Comprehensive Evaluation**: Metrics, visualizations, and model comparison utilities
- **Cross-validation**: Support for k-fold cross-validation

### 4. Deployment & Serving

- **Containerization**: Docker-based deployment for consistent environments
- **Cloud Run**: Scalable, serverless model serving
- **REST API**: Flask-based API for real-time predictions
- **Batch Inference**: Support for batch prediction jobs

### 5. CI/CD & MLOps

- **GitHub Actions**: Automated testing and deployment workflow
- **Artifact Registry**: Version control for container images
- **Model Registry**: Tracking model versions and metadata

## Technical Stack

- **Cloud Platform**: Google Cloud Platform (GCP)
- **ML Framework**: TensorFlow 2.x, TensorFlow Extended (TFX)
- **Data Storage**: BigQuery, Cloud Storage
- **Orchestration**: Vertex AI Pipelines
- **Serving**: Cloud Run, Flask, Gunicorn
- **Development**: Python 3.9, Docker

## Project Structure

```
enterprise-ml-pipeline-gcp/
├── src/                      # Source code
│   ├── data/                 # Data processing modules
│   ├── models/               # Model definition and evaluation
│   ├── pipeline/             # TFX pipeline components
│   └── serving/              # Model serving code
├── notebooks/                # Jupyter notebooks for demos
├── scripts/                  # Utility scripts
├── tests/                    # Unit tests
├── docs/                     # Documentation
│   └── images/               # Architecture diagrams
├── Dockerfile                # Container definition
├── requirements.txt          # Python dependencies
└── README.md                 # Project overview
```

## Key Features

1. **Production-Quality Code**: Well-structured, documented, and tested codebase
2. **Automated Data Validation**: Ensures data quality and consistency
3. **Reproducible Preprocessing**: Transform components for consistent feature engineering
4. **Comprehensive Evaluation**: Metrics and visualizations for model performance
5. **Scalable Serving**: Cloud Run provides auto-scaling for prediction requests
6. **CI/CD Integration**: GitHub Actions for continuous integration and deployment

## Usage Examples

### Running the Pipeline Locally

```bash
python src/pipeline/run_pipeline.py \
    --project_id=your-gcp-project \
    --region=us-central1 \
    --mode=local
```

### Deploying to Vertex AI

```bash
python scripts/submit_pipeline_to_vertex.py \
    --project_id=your-gcp-project \
    --region=us-central1
```

### Deploying the Model to Cloud Run

```bash
bash scripts/deploy_to_cloud_run.sh \
    your-gcp-project \
    us-central1 \
    ./models/serving_model
```

### Making Predictions

```bash
python scripts/model_inference_example.py \
    --model_path=./models/serving_model \
    --num_samples=5
```

## Future Improvements

1. **Hyperparameter Tuning**: Add automated hyperparameter optimization
2. **Model Monitoring**: Implement drift detection and monitoring
3. **Feature Store**: Integrate with a feature store for feature reuse
4. **A/B Testing**: Add support for model experimentation
5. **Explainability**: Implement model explanation tools
6. **Advanced Security**: Add authentication and encryption for the API

## Conclusion

This Enterprise ML Pipeline MVP demonstrates professional ML engineering practices and provides a solid foundation for building production-grade machine learning systems. The modular design allows for easy extension and customization to meet specific business requirements.
