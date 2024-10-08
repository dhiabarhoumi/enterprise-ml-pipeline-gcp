# Enterprise ML Pipeline Architecture

## Architecture Diagram

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

## Component Descriptions

### Data Sources
- **CSV Files**: Local or Cloud Storage data files
- **BigQuery**: Enterprise data warehouse for large-scale data

### TFX Pipeline Components
- **ExampleGen**: Ingests and splits the dataset
- **StatisticsGen**: Computes statistics for data
- **SchemaGen**: Infers schema from statistics
- **ExampleValidator**: Validates data against schema
- **Transform**: Feature engineering using TensorFlow Transform
- **Trainer**: Trains the model using TensorFlow
- **Evaluator**: Evaluates model performance
- **Pusher**: Deploys model to serving infrastructure

### Deployment
- **Vertex AI Registry**: Manages model versions and metadata
- **Cloud Run Container**: Serverless container for model serving

### Client Applications
- **REST API Clients**: Applications consuming the model API
- **Batch Prediction**: Offline prediction for large datasets

## Data Flow

1. Data is sourced from CSV files or BigQuery
2. TFX ExampleGen ingests and splits the data
3. Data is validated and transformed
4. Model is trained and evaluated
5. If model passes quality thresholds, it's pushed to serving
6. Model is deployed as a container on Cloud Run
7. Client applications consume the model via REST API

## Key Features

- **Automated Data Validation**: Ensures data quality and consistency
- **Reproducible Preprocessing**: Transform components ensure consistent feature engineering
- **Model Evaluation**: Comprehensive metrics and validation before deployment
- **Scalable Serving**: Cloud Run provides auto-scaling for prediction requests
- **CI/CD Integration**: GitHub Actions workflow for continuous integration and deployment
