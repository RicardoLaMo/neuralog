# NeuraLog Deployment Guide

This guide covers deploying NeuraLog in various environments (development, staging, production).

## Table of Contents

- [Local Development](#local-development)
- [Docker Deployment](#docker-deployment)
- [Kubernetes Deployment](#kubernetes-deployment)
- [Cloud Platforms](#cloud-platforms)
- [Configuration Management](#configuration-management)
- [Monitoring and Logging](#monitoring-and-logging)
- [Performance Tuning](#performance-tuning)

---

## Local Development

### Quick Start

```bash
# Clone repository
git clone https://github.com/RicardoLaMo/neuralog.git
cd neuralog

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install with development dependencies
pip install -e ".[dev]"

# Set environment variables
cp .env.example .env
# Edit .env with your configuration

# Run tests
pytest

# Run example
python examples/simple_extraction.py
```

### IDE Setup

**VS Code:**
```json
{
  "python.defaultInterpreterPath": "${workspaceFolder}/venv/bin/python",
  "python.linting.enabled": true,
  "python.linting.ruffEnabled": true,
  "[python]": {
    "editor.formatOnSave": true,
    "editor.defaultFormatter": "ms-python.black-formatter"
  }
}
```

**PyCharm:**
- Mark `neuralog/` as Sources Root
- Configure Python interpreter to use venv
- Enable pytest for test discovery

---

## Docker Deployment

### Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    openjdk-11-jre-headless \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir -e "."

# Expose API port (when REST API is implemented)
EXPOSE 8000

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV NEURALOG_LOG_LEVEL=INFO

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python -c "from neuralog.core.engine import Engine; Engine()"

# Entry point
CMD ["python", "-c", "from neuralog.core.engine import Engine; print('NeuraLog ready')"]
```

### Docker Compose

```yaml
version: '3.8'

services:
  neuralog:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: neuralog-app
    environment:
      - NEURALOG_LLM_PROVIDER=openai
      - NEURALOG_LLM_API_KEY=${OPENAI_API_KEY}
      - NEURALOG_ONTOLOGY_PATH=/app/configs/ontologies/default.owl
      - NEURALOG_LOG_LEVEL=INFO
      - NEURALOG_EMBEDDING_DEVICE=cpu
    volumes:
      - ./configs:/app/configs
      - ./data:/app/data
      - ./logs:/app/logs
    ports:
      - "8000:8000"
    networks:
      - neuralog-network

  # Optional: Fuseki Triple Store
  fuseki:
    image: stain/jena-fuseki:latest
    container_name: neuralog-fuseki
    environment:
      - FUSEKI_DATASET_1=neuralog
    ports:
      - "3030:3030"
    volumes:
      - fuseki-data:/fuseki
    networks:
      - neuralog-network

  # Optional: Vector Database
  milvus:
    image: milvusdb/milvus:latest
    container_name: neuralog-milvus
    environment:
      - COMMON_STORAGEQUOTATYPE=hardLimit
    ports:
      - "19530:19530"
    volumes:
      - milvus-data:/var/lib/milvus
    networks:
      - neuralog-network

volumes:
  fuseki-data:
  milvus-data:

networks:
  neuralog-network:
    driver: bridge
```

### Build and Run

```bash
# Build image
docker build -t neuralog:latest .

# Run container
docker run -d --name neuralog \
  -e NEURALOG_LLM_API_KEY=sk-... \
  -p 8000:8000 \
  -v $(pwd)/configs:/app/configs \
  neuralog:latest

# View logs
docker logs -f neuralog

# Using Docker Compose
docker-compose up -d
docker-compose logs -f neuralog

# Stop
docker-compose down
```

---

## Kubernetes Deployment

### ConfigMap

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: neuralog-config
  namespace: default
data:
  config.yaml: |
    llm:
      provider: openai
      model: gpt-4
    ontology:
      reasoner: elk
      consistency_check: true
    embedding:
      device: cuda
      batch_size: 32
    storage:
      triple_store: fuseki
      vector_store: milvus
```

### Secret

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: neuralog-secrets
  namespace: default
type: Opaque
stringData:
  openai-api-key: sk-...
  anthropic-api-key: sk-...
```

### Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: neuralog
  namespace: default
spec:
  replicas: 2
  selector:
    matchLabels:
      app: neuralog
  template:
    metadata:
      labels:
        app: neuralog
    spec:
      containers:
      - name: neuralog
        image: neuralog:latest
        imagePullPolicy: IfNotPresent
        ports:
        - name: api
          containerPort: 8000
        env:
        - name: NEURALOG_LLM_PROVIDER
          value: "openai"
        - name: NEURALOG_LLM_API_KEY
          valueFrom:
            secretKeyRef:
              name: neuralog-secrets
              key: openai-api-key
        - name: NEURALOG_EMBEDDING_DEVICE
          value: "cuda"
        - name: NEURALOG_LOG_LEVEL
          value: "INFO"
        volumeMounts:
        - name: config-volume
          mountPath: /app/configs
        - name: data-volume
          mountPath: /app/data
        - name: logs-volume
          mountPath: /app/logs
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
            nvidia.com/gpu: "1"
          limits:
            memory: "8Gi"
            cpu: "4"
            nvidia.com/gpu: "1"
        livenessProbe:
          exec:
            command:
            - python
            - -c
            - "from neuralog.core.engine import Engine; Engine()"
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          exec:
            command:
            - python
            - -c
            - "from neuralog.core.engine import Engine; Engine()"
          initialDelaySeconds: 10
          periodSeconds: 5
      volumes:
      - name: config-volume
        configMap:
          name: neuralog-config
      - name: data-volume
        emptyDir: {}
      - name: logs-volume
        emptyDir: {}
```

### Service

```yaml
apiVersion: v1
kind: Service
metadata:
  name: neuralog-service
  namespace: default
spec:
  type: LoadBalancer
  ports:
  - port: 80
    targetPort: 8000
    protocol: TCP
    name: api
  selector:
    app: neuralog
```

### Deploy

```bash
# Apply configurations
kubectl apply -f configmap.yaml
kubectl apply -f secret.yaml
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml

# Check status
kubectl get pods -l app=neuralog
kubectl describe deployment neuralog
kubectl logs deployment/neuralog

# Scale
kubectl scale deployment neuralog --replicas=3

# Rolling update
kubectl set image deployment/neuralog neuralog=neuralog:v1.1.0
```

---

## Cloud Platforms

### AWS ECS

Create `ecs-task-definition.json`:

```json
{
  "family": "neuralog",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "2048",
  "memory": "4096",
  "containerDefinitions": [
    {
      "name": "neuralog",
      "image": "neuralog:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "hostPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "NEURALOG_LLM_PROVIDER",
          "value": "openai"
        }
      ],
      "secrets": [
        {
          "name": "NEURALOG_LLM_API_KEY",
          "valueFrom": "arn:aws:secretsmanager:..."
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/neuralog",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```

Deploy:
```bash
aws ecs register-task-definition --cli-input-json file://ecs-task-definition.json
aws ecs run-task --cluster my-cluster --task-definition neuralog
```

### Google Cloud Run

```bash
# Build image
docker build -t gcr.io/my-project/neuralog:latest .

# Push to GCR
docker push gcr.io/my-project/neuralog:latest

# Deploy to Cloud Run
gcloud run deploy neuralog \
  --image gcr.io/my-project/neuralog:latest \
  --platform managed \
  --region us-central1 \
  --memory 4Gi \
  --cpu 2 \
  --set-env-vars NEURALOG_LLM_PROVIDER=openai \
  --set-env-vars NEURALOG_LLM_API_KEY=${OPENAI_API_KEY}
```

### Azure Container Instances

```bash
# Create container group
az container create \
  --resource-group my-group \
  --name neuralog \
  --image neuralog:latest \
  --memory 4 \
  --cpu 2 \
  --environment-variables \
    NEURALOG_LLM_PROVIDER=openai \
    NEURALOG_LLM_API_KEY=$OPENAI_API_KEY \
  --ports 8000
```

---

## Configuration Management

### Environment Variables

```bash
# LLM Configuration
export NEURALOG_LLM_PROVIDER=openai
export NEURALOG_LLM_MODEL=gpt-4
export NEURALOG_LLM_TEMPERATURE=0.7
export NEURALOG_LLM_MAX_TOKENS=2048
export NEURALOG_LLM_API_KEY=sk-...

# Ontology Configuration
export NEURALOG_ONTOLOGY_PATH=/path/to/ontology.owl
export NEURALOG_ONTOLOGY_REASONER=elk

# Embedding Configuration
export NEURALOG_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
export NEURALOG_EMBEDDING_DEVICE=cuda
export NEURALOG_EMBEDDING_BATCH_SIZE=32

# Storage Configuration
export NEURALOG_STORAGE_TRIPLE_STORE=fuseki
export NEURALOG_STORAGE_VECTOR_STORE=milvus
export NEURALOG_STORAGE_DOCUMENT_STORE=elasticsearch

# Logging
export NEURALOG_LOG_LEVEL=INFO
export NEURALOG_LOG_FILE=/var/log/neuralog/app.log
```

### Configuration File (config.yaml)

```yaml
llm:
  provider: openai
  model: gpt-4
  temperature: 0.7
  max_tokens: 2048
  api_key: ${OPENAI_API_KEY}

ontology:
  path: /app/configs/ontologies/biomedical.owl
  reasoner: elk
  consistency_check: true

embedding:
  model: sentence-transformers/all-MiniLM-L6-v2
  device: cuda
  batch_size: 32

verification:
  solver: z3
  soundness_target: 0.99

storage:
  triple_store: fuseki
  triple_store_url: http://fuseki:3030/neuralog
  vector_store: milvus
  vector_store_url: http://milvus:19530
  document_store: elasticsearch
  document_store_url: http://elasticsearch:9200

logging:
  level: INFO
  format: json
  file: /var/log/neuralog/app.log
  max_bytes: 104857600  # 100MB
  backup_count: 10
```

---

## Monitoring and Logging

### Structured Logging

```python
from neuralog.utils.logger import get_logger
import json

logger = get_logger(__name__)

# Log with context
logger.info("Extraction started", extra={
    "text_length": len(text),
    "model": engine.config.llm.model
})

# Log errors with traceback
try:
    result = engine.extract(text)
except Exception as e:
    logger.error("Extraction failed", exc_info=True, extra={
        "error_type": type(e).__name__
    })
```

### Metrics Collection

```python
# Using Prometheus
from prometheus_client import Counter, Histogram, start_http_server
import time

extraction_counter = Counter('neuralog_extractions_total', 'Total extractions')
extraction_duration = Histogram('neuralog_extraction_duration_seconds', 'Extraction duration')

@extraction_duration.time()
def extract_with_metrics(engine, text):
    result = engine.extract(text)
    extraction_counter.inc()
    return result

# Start metrics server
if __name__ == "__main__":
    start_http_server(8001)  # Prometheus scrapes from :8001/metrics
```

### Log Aggregation

**Using ELK Stack:**

```yaml
# docker-compose.yml
elasticsearch:
  image: docker.elastic.co/elasticsearch/elasticsearch:8.0.0
  environment:
    - discovery.type=single-node

kibana:
  image: docker.elastic.co/kibana/kibana:8.0.0
  ports:
    - "5601:5601"

logstash:
  image: docker.elastic.co/logstash/logstash:8.0.0
  volumes:
    - ./logstash.conf:/usr/share/logstash/pipeline/logstash.conf
```

```conf
# logstash.conf
input {
  file {
    path => "/app/logs/*.log"
    codec => json
  }
}

output {
  elasticsearch {
    hosts => ["elasticsearch:9200"]
    index => "neuralog-%{+YYYY.MM.dd}"
  }
}
```

---

## Performance Tuning

### GPU Acceleration

```python
from neuralog.core.config import EmbeddingConfig

# Use GPU for embeddings
config = Config(
    embedding=EmbeddingConfig(
        device="cuda",           # or "mps" for Apple Silicon
        batch_size=64            # Increase batch size on GPU
    )
)
```

### Caching

```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_embedding(text: str):
    return embedding_model.embed(text)
```

### Batch Processing

```python
# Process multiple documents efficiently
def batch_extract(engine, documents: list[str], batch_size: int = 10):
    results = []
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i+batch_size]
        batch_results = engine.distiller.distill_batch(batch)
        results.extend(batch_results)
    return results
```

### Resource Limits

```yaml
# Kubernetes
resources:
  requests:
    memory: "4Gi"
    cpu: "2"
  limits:
    memory: "8Gi"
    cpu: "4"
```

```dockerfile
# Docker
RUN pip install --no-cache-dir \
    psutil \
    memory-profiler

# Monitor memory
CMD ["python", "-m", "memory_profiler", "app.py"]
```

---

## Troubleshooting

### Common Issues

1. **Out of Memory**
   ```bash
   # Reduce batch size
   export NEURALOG_EMBEDDING_BATCH_SIZE=8

   # Use CPU instead of GPU
   export NEURALOG_EMBEDDING_DEVICE=cpu
   ```

2. **LLM API Errors**
   ```python
   from tenacity import retry, stop_after_attempt, wait_exponential

   @retry(stop=stop_after_attempt(3), wait=wait_exponential())
   def safe_extract(engine, text):
       return engine.extract(text)
   ```

3. **Ontology Loading Failures**
   ```python
   # Validate ontology
   logger.info(f"Loaded ontology with {len(ontology.get_entities())} entities")
   logger.info(f"Loaded ontology with {len(ontology.get_relations())} relations")
   ```

### Health Checks

```bash
# Check if service is running
curl -X GET http://localhost:8000/health

# Check resource usage
docker stats neuralog

# Check logs
docker logs neuralog | tail -100
```

---

## References

- [Docker Documentation](https://docs.docker.com/)
- [Kubernetes Docs](https://kubernetes.io/docs/)
- [AWS ECS](https://aws.amazon.com/ecs/)
- [Google Cloud Run](https://cloud.google.com/run)
