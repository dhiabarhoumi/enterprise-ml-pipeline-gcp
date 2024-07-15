# Enterprise ML Pipeline on Google Cloud Platform

## Project Summary

This project implements a production-quality, end-to-end machine learning pipeline using Google Cloud Platform's Vertex AI for automated model training and deployment. The solution demonstrates a professional MLOps workflow that handles data validation, preprocessing, model training, evaluation, and deployment with a focus on scalability, maintainability, and efficiency.

The pipeline leverages TensorFlow Extended (TFX) for data validation, preprocessing, and model evaluation, while using BigQuery for efficient data storage and feature engineering. The entire workflow is containerized using Docker and deployed with Cloud Run for scalable, serverless model serving.

For a detailed development timeline, see [Project Timeline](docs/project_timeline.md).

## Features

- **End-to-end ML pipeline** using Google Cloud Platform's Vertex AI Pipelines
- **Automated data validation** and schema inference with TensorFlow Data Validation
- **Reproducible preprocessing** with TensorFlow Transform for consistent feature engineering
- **Efficient feature engineering** with optimized BigQuery data pipelines
- **Model training and evaluation** with comprehensive performance metrics tracking
- **Containerized deployment** using Docker and Cloud Run for serverless scaling
- **REST API for model serving** with both single and batch prediction endpoints
- **CI/CD integration** with GitHub Actions for automated testing and deployment
- **Synthetic data generation** for demonstration and testing purposes
- **Comprehensive documentation** and demo notebook for easy onboarding

## Architecture

The architecture diagram below illustrates the components and data flow of the ML pipeline:

```
+----------------------------------+
|                                  |
|  Data Sources                    |
|  +------------+  +------------+  |
|  | CSV Files  |  | BigQuery   |  |
|  +------------+  +------------+  |
|         |             |         |
+---------+-------------+---------+
          |             |
          v             v
+----------------------------------+
|                                  |
|  TFX Pipeline                    |
|  +------------+  +------------+  |
|  | ExampleGen |->| StatsGen   |  |
|  +------------+  +------------+  |
|                        |         |
|                        v         |
|  +------------+  +------------+  |
|  | SchemaGen  |<-| ExampleVal |  |
|  +------------+  +------------+  |
|        |                |       |
|        v                v       |
|  +------------+  +------------+  |
|  | Transform  |->| Trainer    |  |
|  +------------+  +------------+  |
|                       |         |
|                       v         |
|  +------------+  +------------+  |
|  | Evaluator  |<-| Pusher     |  |
|  +------------+  +------------+  |
|                       |         |
+----------------------------------+
                      |
                      v
+----------------------------------+
|                                  |
|  Deployment                      |
|  +------------+  +------------+  |
|  | Vertex AI  |  | Cloud Run  |  |
|  | Registry   |->| Container  |  |
|  +------------+  +------------+  |
|                       |         |
+----------------------------------+
                      |
                      v
+----------------------------------+
|                                  |
|  Client Applications             |
|  +------------+  +------------+  |
|  | REST API   |  | Batch      |  |
|  | Clients    |  | Prediction |  |
|  +------------+  +------------+  |
|                                  |
+----------------------------------+
```

For more details on the architecture, see [docs/images/architecture.md](docs/images/architecture.md).

## Quickstart

### Prerequisites

- Google Cloud Platform account with billing enabled
- Python 3.11+ (recommended, 3.9+ minimum)
- Docker
- Google Cloud SDK (gcloud CLI)
- Git

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/enterprise-ml-pipeline-gcp.git
   cd enterprise-ml-pipeline-gcp
   ```

2. Set up the environment automatically using the provided script:
   ```bash
   # On Linux/Mac
   ./scripts/setup_environment.sh your-gcp-project-id us-central1
   
   # On Windows
   scripts\setup_environment.sh your-gcp-project-id us-central1
   ```
   
   Or manually set up a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Set up Google Cloud credentials:
   ```bash
   gcloud auth application-default login
   ```

4. Configure your GCP project:
   ```bash
   export PROJECT_ID="your-gcp-project-id"
   export REGION="us-central1"  # Choose your preferred region
   ```

## Example Usage

### 1. Data Preparation

Generate synthetic data and upload to BigQuery:

```bash
python src/data/generate_synthetic_data.py --output_path=data/retail_data.csv --num_samples=10000
python src/data/upload_to_bigquery.py --project_id=$PROJECT_ID --dataset=retail_dataset --table=transactions --data_path=data/retail_data.csv
```

### 2. Validate Data

```bash
python src/data/validate_data.py --project_id=$PROJECT_ID --dataset=retail_dataset --table=transactions
```

### 3. Run the Pipeline Locally

```bash
python src/pipeline/run_pipeline.py --project_id=$PROJECT_ID --region=$REGION --mode=local
```

### 4. Submit the Pipeline to Vertex AI

```bash
python scripts/submit_pipeline_to_vertex.py --project_id=$PROJECT_ID --region=$REGION
```

### 5. Deploy the Model to Cloud Run

```bash
# Using the deployment script
bash scripts/deploy_to_cloud_run.sh $PROJECT_ID $REGION ./models/serving_model

# Or using the model deployment script
bash scripts/deploy_model.sh $PROJECT_ID $REGION
```

### 6. Make Predictions

```bash
python scripts/model_inference_example.py --model_path=./models/serving_model --num_samples=5
```

## Demo

A comprehensive Jupyter notebook demonstrating the pipeline's capabilities is available at `notebooks/pipeline_demo.ipynb`. This notebook walks through:

1. Environment setup and configuration
2. Synthetic data generation and upload to BigQuery
3. Data validation and schema inference
4. Running the TFX pipeline locally
5. Model evaluation and performance metrics
6. Making predictions with the trained model
7. Deploying the model to Cloud Run

## Project Structure

```
enterprise-ml-pipeline-gcp/
├── docs/
│   └── images/
│       ├── architecture.md
│       └── architecture.png
│   └── project_summary.md
├── notebooks/
│   └── pipeline_demo.ipynb
├── scripts/
│   ├── deploy_model.sh
│   ├── deploy_to_cloud_run.sh
│   ├── model_inference_example.py
│   ├── setup_environment.sh
│   └── submit_pipeline_to_vertex.py
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── bigquery_utils.py
│   │   ├── data_validation.py
│   │   ├── generate_synthetic_data.py
│   │   └── upload_to_bigquery.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── model.py
│   │   └── evaluation.py
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── components.py
│   │   ├── config.py
│   │   ├── run_pipeline.py
│   │   ├── transform.py
│   │   └── trainer.py
│   └── serving/
│       ├── __init__.py
│       ├── app.py
│       └── prediction.py
├── tests/
│   ├── __init__.py
│   ├── test_bigquery_utils.py
│   ├── test_data_validation.py
│   ├── test_model_evaluation.py
│   └── test_pipeline_components.py
├── .github/
│   └── workflows/
│       └── ci-cd.yml
├── .gitignore
├── Dockerfile
├── LICENSE
├── README.md
└── requirements.txt
```

## Future Improvements

1. **Feature Store Integration**: Implement Vertex AI Feature Store for better feature reuse, versioning, and serving.

2. **Automated Retraining**: Set up scheduled retraining based on data drift detection to ensure model performance over time.

3. **A/B Testing Framework**: Develop a framework for comparing model versions in production with statistical significance testing.

4. **Enhanced Monitoring**: Implement comprehensive monitoring for model performance, data drift, and prediction quality in production.

5. **Hyperparameter Tuning**: Add automated hyperparameter optimization using Vertex AI's Vizier service.

6. **Model Explainability**: Integrate TensorFlow Model Analysis (TFMA) and What-If Tool for model interpretability.

7. **Advanced Security**: Implement VPC Service Controls, Private Service Connect, and fine-grained IAM for enhanced security.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

Your Name - your.email@example.com

## Acknowledgments

- Google Cloud Platform documentation and examples
- TensorFlow Extended (TFX) team for their excellent framework
- The open-source ML community for inspiration and best practices

## Disclaimer

This is a production-quality MVP created for demonstration and job application purposes. While it implements comprehensive functionality following industry best practices, some aspects might require further customization for specific enterprise use cases.
