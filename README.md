# 🛡️ Phishing Website Classifier

> **Advanced Machine Learning Solution for Cybersecurity Threat Detection**

A sophisticated neural network-based system that identifies and classifies malicious URLs using advanced feature extraction and deep learning techniques. This project demonstrates expertise in machine learning, cybersecurity, and full-stack web development.

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-2.0+-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![NumPy](https://img.shields.io/badge/NumPy-1.21+-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)
[![Ask DeepWiki](https://img.shields.io/badge/Ask-DeepWiki-9B59B6?style=for-the-badge&logo=wikipedia&logoColor=white)](https://deepwiki.com/KIKW12/Phishing_Website_Classifier)

## 🎯 Project Overview

This project tackles the critical cybersecurity challenge of phishing detection through machine learning. With the exponential rise in phishing attacks (over 3.4 billion phishing emails sent daily), automated detection systems are essential for protecting users and organizations from cyberthreats.

**Key Achievement**: Achieved **73.5% accuracy** in detecting malicious URLs using a custom-built neural network with advanced feature engineering.

## 🚀 Features

### 🔍 **Advanced Feature Extraction**
- **IP Address Detection**: Identifies URLs using IP addresses instead of domain names
- **Character Pattern Analysis**: Analyzes suspicious characters (@, -, ~, //) in URLs
- **Domain Analysis**: Detects embedded domains and suspicious TLDs
- **URL Length Analysis**: Statistical analysis of URL length patterns
- **Shortening Service Detection**: Identifies known URL shortening services
- **HTTPS Token Analysis**: Detects deceptive HTTPS usage in domain names

### 🧠 **Custom Neural Network**
- **Architecture**: Multi-layer perceptron with ReLU and Sigmoid activations
- **Advanced Techniques**: 
  - He initialization for optimal weight distribution
  - Momentum-based gradient descent for faster convergence
  - Mini-batch training for improved performance
  - Feature standardization for stable learning
- **Performance Optimization**: Best model checkpointing and early stopping

### 🌐 **Web Application**
- **Real-time Classification**: Instant URL analysis through REST API
- **Modern UI**: Responsive design with Tailwind CSS
- **Production Ready**: Gunicorn WSGI server configuration
- **Cloud Deployment**: Heroku-ready with Procfile configuration

## 📊 Technical Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   URL Input     │───▶│ Feature Extractor │───▶│ Neural Network  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
                    ┌──────────────────┐    ┌─────────────────┐
                    │ 10 Key Features  │    │ Binary Output   │
                    │ • IP Usage       │    │ • Safe/Malicious│
                    │ • Character Count│    │ • Confidence    │
                    │ • Domain Analysis│    └─────────────────┘
                    │ • URL Length     │
                    └──────────────────┘
```

## 🛠️ Technology Stack

### **Backend & ML**
- **Python 3.8+**: Core programming language
- **NumPy**: Numerical computations and matrix operations
- **Pandas**: Data manipulation and analysis
- **Flask**: Web framework for API development
- **Custom Neural Network**: Built from scratch using NumPy

### **Frontend**
- **HTML5/CSS3**: Modern web standards
- **JavaScript (ES6+)**: Interactive user interface
- **Tailwind CSS**: Utility-first CSS framework

### **Data Sources**
- **PhishTank**: Real-time phishing data
- **Kaggle Datasets**: 
  - Malicious URLs dataset
  - Phishing site URLs dataset

### **Deployment**
- **Gunicorn**: WSGI HTTP Server
- **Heroku**: Cloud platform deployment

## 📈 Performance Metrics

| Metric | Value |
|--------|-------|
| **Accuracy** | 73.5% |
| **Training Epochs** | 100 |
| **Features** | 10 engineered features |
| **Dataset Size** | 120,000+ URLs |
| **Model Architecture** | 10→11→1 neurons |

## 🔧 Installation & Setup

### Prerequisites
```bash
python >= 3.8
pip >= 21.0
```

### Quick Start
```bash
# Clone the repository
git clone https://github.com/KIKW12Phishing_Website_Classifier.git
cd Phishing_Website_Classifier

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py
```

### Docker Setup (Optional)
```bash
# Build Docker image
docker build -t phishing-classifier .

# Run container
docker run -p 5000:5000 phishing-classifier
```

## 🎮 Usage

### Web Interface
1. Navigate to `http://localhost:5000`
2. Enter a URL in the input field
3. Click "Classify" to get instant results
4. View the classification result and confidence score

### API Endpoint
```bash
curl -X POST http://localhost:5000/classify \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com"}'
```

**Response:**
```json
{
  "url": "https://example.com",
  "is_malicious": false,
  "features": {
    "use_of_ip": 0,
    "count.": 1,
    "url_length": 19,
    "...": "..."
  }
}
```

## 🧪 Model Training Process

### 1. Data Preprocessing
- Label encoding for categorical variables
- Feature extraction from raw URLs
- Dataset splitting (train/test/validation)

### 2. Feature Engineering
10 carefully selected features based on cybersecurity research:
- IP address usage indicator
- Special character counts (., @, -, ~)
- Embedded domain detection
- Suspicious keyword presence
- URL shortening service detection
- HTTPS token in domain
- URL length analysis

### 3. Neural Network Training
- **Input Layer**: 10 features
- **Hidden Layer**: 11 neurons with ReLU activation
- **Output Layer**: 1 neuron with Sigmoid activation
- **Optimization**: Adam optimizer with momentum
- **Loss Function**: Binary cross-entropy

## 📚 Key Learning Outcomes

This project demonstrates proficiency in:

### **Machine Learning**
- Custom neural network implementation from scratch
- Advanced optimization techniques (momentum, mini-batch)
- Feature engineering for cybersecurity applications
- Model evaluation and performance tuning

### **Software Engineering**
- RESTful API design and implementation
- Full-stack web development
- Code organization and modularity
- Version control with Git

### **Cybersecurity**
- Understanding of phishing attack vectors
- URL analysis and threat detection
- Security-focused feature engineering
- Real-world application of ML in cybersecurity

### **Data Science**
- Large dataset handling and preprocessing
- Statistical analysis and visualization
- Model validation and testing
- Performance metrics evaluation

## 🔬 Research & Development

### Dataset Analysis
- **68.4%** legitimate URLs
- **31.6%** phishing URLs
- Comprehensive feature analysis across multiple threat categories

### Feature Importance
Based on cybersecurity research and statistical analysis:
1. URL length patterns
2. IP address usage
3. Suspicious character presence
4. Domain embedding techniques
5. HTTPS deception tactics

## 🚀 Future Enhancements

- [ ] **Deep Learning**: Implement CNN/LSTM for URL sequence analysis
- [ ] **Real-time Updates**: Automatic model retraining with new threat data
- [ ] **API Rate Limiting**: Enhanced security for production deployment
- [ ] **Batch Processing**: Support for bulk URL analysis
- [ ] **Mobile App**: React Native mobile application
- [ ] **Browser Extension**: Real-time protection while browsing

## 📄 Project Structure

```
Phishing_Website_Classifier/
├── app.py                      # Flask web application
├── improved_nn.py              # Custom neural network implementation
├── feature_extraction.ipynb    # Feature engineering notebook
├── phishing_classification.ipynb # Model training notebook
├── requirements.txt            # Python dependencies
├── Procfile                   # Heroku deployment config
├── templates/
│   └── index.html             # Web interface
├── *.npy                      # Trained model weights
└── df_*.csv                   # Processed datasets
```
### 🌟 **"Transforming cybersecurity through intelligent automation and machine learning innovation"**

---

*This project represents a comprehensive approach to solving real-world cybersecurity challenges through advanced machine learning techniques, demonstrating both technical expertise and practical application in protecting digital assets.*
