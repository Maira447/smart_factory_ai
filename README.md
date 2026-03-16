
# 🧠 SmartWeave AI

### Intelligent AI Wrapper for Smart Manufacturing & Textile Defect Detection

SmartWeave AI is an **AI-powered industrial decision platform** designed to help manufacturing industries detect product defects, predict machine failures, and optimize production using advanced machine learning models.

The system integrates **multiple AI models into a single AI wrapper architecture**, enabling factories to move from **manual inspection and reactive maintenance** to **predictive and intelligent manufacturing**.

---

# 🚨 Problem Statement

Many manufacturing industries, particularly **textile factories in developing economies**, still rely on:

* Manual fabric inspection
* Reactive machine maintenance
* Limited production analytics
* Human-dependent quality control

These limitations cause:

* Product defects reaching final production
* Unexpected machine failures
* Production downtime
* Financial losses due to rejected exports

Despite generating large amounts of data, factories **lack AI systems that convert data into actionable decisions**.

---

# 💡 Proposed Solution

SmartWeave AI introduces an **AI Wrapper Architecture** that integrates multiple intelligent models to support industrial decision-making.

The system contains three AI modules:

1. **Textile Defect Detection**
2. **Predictive Maintenance (Machine Failure Prediction)**
3. **Production Forecasting**

All modules work together through a **central AI wrapper layer**, allowing factories to monitor quality, maintenance, and production from a single platform.

---

# 🧩 System Architecture

```
                Industrial Data Sources
        -------------------------------------
        | Textile Images | Sensor Data | Logs |
        -------------------------------------
                      │
                      ▼
               AI Wrapper Layer
        --------------------------------
        | CNN / EfficientNet (Vision) |
        | XGBoost (Maintenance)       |
        | LSTM (Forecasting)          |
        --------------------------------
                      │
                      ▼
             Industrial Decision Dashboard
        --------------------------------
        | Defect Detection Results     |
        | Machine Failure Alerts       |
        | Production Forecast Reports  |
        --------------------------------
```

---

# 📊 Project Modules

## 1️⃣ Textile Defect Detection

Detects defects in textile fabric images using **deep learning models**.

### Dataset

* Textile H5 image dataset
* CSV metadata files

### Features

* Image preprocessing
* Label mapping using CSV
* CNN-based defect classification

### Models

* CNN
* EfficientNet (optional upgrade)

### Output

```
Prediction: Hole Defect
Confidence: 94%
```

---

## 2️⃣ Predictive Maintenance

Predicts machine failures before they occur using sensor data.

### Dataset

Machine sensor readings including:

* temperature
* vibration
* pressure
* machine load

### Techniques

* SMOTE for class imbalance
* Feature engineering
* XGBoost classifier

### Output

```
Failure Risk: 78% in next 24 hours
```

---

## 3️⃣ Production Forecasting

Predicts future production output based on historical manufacturing data.

### Model

* LSTM (Long Short-Term Memory)

### Purpose

* Production planning
* Demand forecasting
* Resource allocation

### Output

```
Predicted Output (Next Week): 520 Units
```

---

# ⚙️ Technologies Used

| Category         | Tools                 |
| ---------------- | --------------------- |
| Machine Learning | Scikit-learn, XGBoost |
| Deep Learning    | TensorFlow, Keras     |
| Data Processing  | Pandas, NumPy         |
| Image Processing | OpenCV                |
| Visualization    | Matplotlib            |
| Storage          | HDF5 (h5py)           |
| Deployment       | Streamlit / FastAPI   |

---

# 📂 Project Structure

```
SmartWeave-AI/
│
├── datasets/
│   ├── train32.h5
│   ├── train64.h5
│   ├── test32.h5
│   ├── test64.h5
│   ├── train32.csv
│   ├── train64.csv
│   ├── test32.csv
│   └── test64.csv
│
├── models/
│   ├── textile_defect_model.h5
│   ├── maintenance_model.pkl
│   └── forecasting_model.h5
│
├── notebooks/
│   ├── phase1_data_analysis.ipynb
│   ├── phase2_model_training.ipynb
│
├── app/
│   ├── streamlit_app.py
│
├── utils/
│   ├── data_loader.py
│   ├── preprocessing.py
│
└── README.md
```

---

# 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/SmartWeave-AI.git
cd SmartWeave-AI
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# 🧪 Running Model Training

Run the training notebook:

```
notebooks/phase2_model_training.ipynb
```

This will:

* Load H5 datasets
* Map labels using CSV
* Train CNN models
* Save trained models

---

# 🖥 Running the Application

Start the web interface:

```bash
streamlit run app/streamlit_app.py
```

Users can:

* Upload textile images
* Detect defects
* View AI predictions

---

# 📈 Model Evaluation

Metrics used:

| Module                 | Evaluation Metrics          |
| ---------------------- | --------------------------- |
| Defect Detection       | Accuracy, Precision, Recall |
| Predictive Maintenance | ROC-AUC                     |
| Forecasting            | MAE, RMSE                   |

---

# 🌍 Impact

SmartWeave AI supports the transition toward **Industry 4.0**, especially in developing economies where industrial automation is limited.

Benefits include:

* Reduced product defects
* Early machine failure detection
* Improved production planning
* Lower operational losses

---

# 🔮 Future Work

* Real-time IoT sensor integration
* Multi-factory monitoring dashboard
* Edge AI deployment
* Automated model retraining pipelines

---

# 👩‍💻 Author

**Maira Zafar**
AI & Machine Learning Enthusiast
Computer Systems Engineering Student

Interests:

* AI for Industry
* Machine Learning Systems
* Smart Manufacturing
* AI-driven Automation

---

