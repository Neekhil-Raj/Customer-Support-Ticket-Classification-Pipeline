# Customer Support Ticket Classification Pipeline

A full-stack, end-to-end machine learning solution for automating the classification, routing, and summarization of customer support tickets. Leverage classical ML, deep learning and LLMs, with real-time ingestion, MLOps tooling, and production-ready deployment.

## Features

- Scalable ETL pipelines for CSV, Kafka streams, and image-based tickets (OCR)
- Text preprocessing with spaCy, NLTK, TF-IDF; image text extraction via OpenCV & Tesseract
- Traditional ML models (Random Forest, SVM, Logistic Regression) and hyperparameter tuning
- Deep learning (LSTM, CNN, hybrid text+numerical) with TensorFlow and Keras
- LLM integration for classification, summarization, and routing (OpenAI GPT & Hugging Face Zero-Shot)
- Workflow orchestration with Apache Airflow; experiment tracking with MLflow
- Real-time streaming via Kafka; dashboards in Tableau or Streamlit
- REST API built with FastAPI and Dockerized microservice architecture
- CI/CD with GitHub Actions; deployment via Docker Compose or Kubernetes

## Table of Contents

1. [Requirements](#requirements)  
2. [Installation](#installation)  
3. [Configuration](#configuration)  
4. [Data Preparation](#data-preparation)  
5. [Running Locally](#running-locally)  
6. [Docker-Compose Deployment](#docker-compose-deployment)  
7. [Usage](#usage)  
8. [Tests](#tests)  
9. [Contributing](#contributing)  
10. [License](#license)

## Requirements

- **Operating System**: Linux, macOS, or Windows 10+  
- **Hardware**:  
  - CPU: 4 cores (8+ recommended)  
  - RAM: 8 GB (16 GB recommended)  
  - Optional NVIDIA GPU (CUDA 11.8+) for deep learning  
- **Software**:  
  - Python 3.10 or 3.11  
  - Git 2.44+  
  - Docker 24.0+ and Docker Compose  
  - (Optional) Conda/Miniconda for virtual environments  

## Installation

### Option A: Local-Bare (Conda + Pip)

1. Clone the repository and navigate into it:
   ```bash
   git clone https://github.com//customer-support-ticket-classification.git
   cd customer-support-ticket-classification
   ```

2. Create a Conda environment:
   ```bash
   conda create -n ticket-classifier python=3.11 -y
   conda activate ticket-classifier
   ```

3. Upgrade core build tools:
   ```bash
   pip install --upgrade pip setuptools wheel
   ```

4. Install GPU-accelerated PyTorch (or CPU-only):
   ```bash
   # GPU (CUDA 11.8)
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   # OR CPU-only
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
   ```

5. Install remaining dependencies:
   ```bash
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   ```

### Option B: Docker-Compose

1. Clone and enter the repo:
   ```bash
   git clone https://github.com//customer-support-ticket-classification.git
   cd customer-support-ticket-classification
   ```

2. Create a `.env` file in the project root with the following variables:
   ```
   MYSQL_HOST=mysql
   MYSQL_USER=root
   MYSQL_PASSWORD=rootpassword
   MYSQL_DATABASE=support_tickets
   MONGODB_URI=mongodb://mongodb:27017
   KAFKA_BOOTSTRAP_SERVERS=kafka:9092
   MLFLOW_TRACKING_URI=http://mlflow:5000
   OPENAI_API_KEY=
   HUGGINGFACE_TOKEN=
   ```

3. Build and start all services:
   ```bash
   docker-compose up --build -d
   ```

4. Initialize the MySQL schema:
   ```bash
   docker exec -it app python - <<'EOF'
   from config.database import DatabaseManager
   db = DatabaseManager()
   conn = db.connect_mysql()
   db.create_mysql_tables()
   EOF
   ```

## Configuration

All environment-specific settings are managed via environment variables or the `config/config.py` file:

- **Database**: `MYSQL_HOST`, `MYSQL_USER`, `MYSQL_PASSWORD`, `MYSQL_DATABASE`, `MONGODB_URI`  
- **Kafka**: `KAFKA_BOOTSTRAP_SERVERS`, `KAFKA_TOPIC`  
- **MLflow**: `MLFLOW_TRACKING_URI`, `MLFLOW_EXPERIMENT_NAME`  
- **OpenAI & HF**: `OPENAI_API_KEY`, `HUGGINGFACE_TOKEN`  
- **API**: `API_HOST`, `API_PORT`, `DEBUG`

## Data Preparation

Place your `customer_support_tickets.csv` in the project root or mount it into `/app/data/`. The CSV must include:

```
Ticket ID, Customer Name, Customer Email, Customer Age, Customer Gender,
Product Purchased, Date of Purchase, Ticket Type, Ticket Subject,
Ticket Description, Ticket Status, Resolution, Ticket Priority,
Ticket Channel, First Response Time, Time to Resolution, Customer Satisfaction Rating
```

## Running Locally

1. **Generate Sample Data** (optional):
   ```bash
   python data/generate_sample_data.py
   ```

2. **Data Exploration**:
   ```bash
   jupyter lab notebooks/01_data_exploration_actual.ipynb
   ```

3. **Train Models**:
   ```bash
   python src/pipeline/training_pipeline_actual.py
   ```

4. **Launch API**:
   ```bash
   uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
   ```

5. **Access Swagger UI**:  
   Open `http://localhost:8000/docs` in your browser.

## Docker-Compose Deployment

Once `docker-compose up` is running:

- MySQL: `localhost:3306`  
- MongoDB: `localhost:27017`  
- Kafka: `localhost:9092`  
- MLflow UI: `http://localhost:5000`  
- FastAPI: `http://localhost:8000`  
- Airflow UI: `http://localhost:8080`  

Use the `/api/v1/data/upload` endpoint to ingest your CSV, then `/api/v1/train/pipeline` to train models. Classify tickets via `/api/v1/tickets/classify`.

## Usage

### Classify a Ticket
```bash
curl -X POST http://localhost:8000/api/v1/tickets/classify \
  -H "Content-Type: application/json" \
  -d '{
    "customer_name":"Jane Doe",
    "customer_email":"jane@example.com",
    "ticket_subject":"Cannot login",
    "ticket_description":"Error 403 on login",
    "ticket_priority":"High",
    "ticket_channel":"Email"
  }'
```

### Upload CSV
```bash
curl -X POST http://localhost:8000/api/v1/data/upload \
  -F "file=@customer_support_tickets.csv"
```

## Tests

Run unit and integration tests with:
```bash
pytest -q
```

## Contributing

1. Fork the repository  
2. Create a feature branch (`git checkout -b feat/my-feature`)  
3. Commit your changes (`git commit -m "feat: description"`)  
4. Push to your branch (`git push origin feat/my-feature`)  
5. Open a Pull Request  

Please follow the code style (Black) and run all tests before submitting.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.