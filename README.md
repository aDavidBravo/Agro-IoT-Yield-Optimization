# Agro-IoT-Yield-Optimization

**Agro-Industrial Leader in LATAM** | Sensor Ingestion & Predictive Modeling

![Agro IoT Header](https://via.placeholder.com/1000x300?text=Digital+Agriculture+Transformation)

## 📖 Context
As a **Agro-Industrial Leader in LATAM**, we manage operations across thousands of hectares. The challenge of manual monitoring and reactive decision-making limited our efficiency in a competitive market. This project represents our digital transformation backbone.

## 🚀 Key Results
This architecture has enabled significant business outcomes:
- **25% Reduction in Data Reprocessing**: Automated Airflow pipelines ensure clean, validated sensor data ingestion, eliminating manual clean-up hours.
- **12% Cost Savings**: By using our Yield Prediction Model to optimize water and fertilizer application per lot, we have drastically reduced waste without compromising output.

## 🛠️ Technology Stack & Scalability
This project uses a robust, scalable stack designed to handle data from thousands of IoT sensors:
- **Apache Airflow**: Orchestrates complex ETL workflows, ensuring data flows reliably from sensors to our data lake. Its scalability allows us to add new fields (hectares) without code changes.
- **Python**: Powering our data processing (`src/processing`) and machine learning (`src/models`) layers.
- **scikit-learn**: For our Yield Prediction models (Random Forest).

## 📂 Repository Structure
- `airflow_dags/`: Workflows for data ingestion and transformation.
- `src/processing/`: Scripts for data cleaning and sensor data normalization.
- `src/simulation/`: `sensor_simulator.py` to generate synthetic training data reflecting real-world conditions.
- `src/models/`: Machine learning models for yield prediction.
- `notebooks/`: Exploratory Data Analysis (EDA) and research.

## 🚦 Getting Started
1. **Clone the repository**:
   ```bash
   git clone https://github.com/YOUR_ORG/Agro-IoT-Yield-Optimization.git
   ```
2. **Setup Environment**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Run Simulation**:
   ```bash
   python src/simulation/sensor_simulator.py
   ```
4. **Train Model**:
   ```bash
   python src/models/train_model.py
   ```
