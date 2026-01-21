# 🌳 Interactive Decision Trees Dashboard

An interactive web application for training, visualizing, and analyzing decision tree classifiers. Built with Streamlit and scikit-learn for data scientists, students, and analysts.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.39+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

**Developed by Álvaro Andrés De Lamo**  
University of Silesia - Decision and Association Rules Course 🇵🇱

---

## ✨ Features

### 🎯 **Core Functionality**
- **Interactive Training**: Configure and train decision trees with customizable hyperparameters
- **Automatic Rule Extraction**: Extract human-readable decision rules from trained trees
- **Visual Tree Diagrams**: Interactive and static visualizations of tree structure
- **Performance Analysis**: Comprehensive metrics including accuracy, precision, recall, and F1 score

### 📊 **Visualizations**
- Decision tree diagram with color-coded nodes
- Confusion matrix heatmap
- Feature importance rankings
- Class distribution charts
- Tree depth and structure analysis
- Train vs. test performance comparison

### 🔧 **Data Processing**
- Automatic handling of missing values (mean/median/mode imputation or dropping)
- Categorical variable encoding
- Intelligent CSV parsing (auto-detect separators and headers)
- Support for both numeric and categorical features

### 📈 **Model Configuration**
- Split criterion (Gini impurity or Entropy)
- Maximum tree depth control
- Minimum samples for splitting nodes
- Minimum samples per leaf
- Train/test split ratio adjustment

### 💾 **Export Capabilities**
- Download decision rules as TXT
- Export predictions to CSV
- Save feature importance rankings
- Generate model summary report

---

## 🚀 Quick Start

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/aadl6-ua/decision-trees-dashboard.git
cd decision-trees-dashboard
```

2. **Create virtual environment (recommended)**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Running the App

```bash
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`

---

## 📖 Usage Guide

### 1. **Load Data**
- **Option A**: Upload your own CSV file via the sidebar
- **Option B**: Use the included example dataset (Cars classification)

**Data Requirements**:
- CSV format (comma or semicolon separated)
- Header row with column names
- At least one feature column and one target column
- Supports numeric and categorical data

### 2. **Configure Preprocessing**
- Select target column to predict
- Choose missing value handling strategy:
  - `mean`: Replace with column mean (numeric)
  - `median`: Replace with column median (numeric)
  - `mode`: Replace with most frequent value
  - `drop`: Remove rows with missing values

### 3. **Set Model Parameters**

**Test Set Size** (0.1 - 0.5):
- Proportion of data reserved for testing
- Default: 0.3 (70% train, 30% test)

**Split Criterion**:
- `gini`: Gini impurity (default, faster)
- `entropy`: Information gain (more computationally intensive)

**Maximum Depth** (1 - 20):
- Limits tree depth to prevent overfitting
- Higher values = more complex trees
- Set to 20 for unlimited depth

**Min Samples to Split** (2 - 20):
- Minimum samples required to split an internal node
- Higher values = simpler trees

**Min Samples in Leaf** (1 - 20):
- Minimum samples required in leaf nodes
- Higher values = smoother decision boundaries

### 4. **Train the Model**
Click **"🚀 Train Decision Tree"** to:
- Preprocess the data
- Train the decision tree classifier
- Generate visualizations
- Extract decision rules
- Calculate performance metrics

### 5. **Analyze Results**
Explore the generated insights:
- **Performance Metrics**: Accuracy, precision, recall, F1 score
- **Confusion Matrix**: Visualize prediction errors
- **Tree Visualization**: See the complete decision tree structure
- **Feature Importance**: Identify most influential features
- **Decision Rules**: Human-readable if-then rules
- **Export Options**: Download results for further analysis

---

## 📂 Project Structure

```
decision-trees-dashboard/
├── app.py                        # Main Streamlit application
├── requirements.txt              # Python dependencies
├── README.md                     # This file
├── .gitignore                   # Git ignore rules
├── utils/
│   ├── __init__.py              # Package initialization
│   ├── preprocessing.py         # Data loading and preprocessing
│   ├── decision_trees.py        # Model training and rule extraction
│   └── visualization.py         # Plotly and Matplotlib visualizations
├── data/
│   └── encoded_data_changed2.csv  # Example dataset (Cars)
└── docs/
    └── original_code.ipynb      # Original Jupyter notebook
```

---

## 🎯 Decision Trees Overview

### What are Decision Trees?

Decision trees are supervised learning algorithms that create a model predicting target values by learning simple decision rules from data features. They:

- **Split data** recursively based on feature values
- **Create branches** representing decision paths
- **Assign class labels** at leaf nodes
- **Generate interpretable rules** that humans can understand

### When to Use Decision Trees?

✅ **Best for**:
- Classification tasks with categorical or numeric features
- Problems requiring interpretable models
- Datasets with non-linear relationships
- Situations where feature interactions matter

⚠️ **Consider alternatives when**:
- Dataset is very large (>100k rows) - use Random Forests
- Features are highly correlated - use regularized models
- Need probabilistic predictions - use Logistic Regression

### Hyperparameter Tuning Tips

- **Start simple**: Begin with default parameters and a small max_depth (4-6)
- **Monitor overfitting**: If train accuracy >> test accuracy, reduce complexity
- **Increase complexity gradually**: Add depth if underfitting
- **Use validation**: Compare multiple configurations on held-out data
- **Prune aggressively**: Higher min_samples_leaf = simpler trees

---

## 📊 Example Dataset

**Cars Classification Dataset** includes:
- Vehicle characteristics (buying price, maintenance cost, number of doors, etc.)
- Safety ratings
- Luggage capacity
- **Target**: Car acceptability class

**Use case**: Classify car acceptability based on features to help consumers make informed decisions.

---

## 🌐 Deployment

### Deploy to Streamlit Cloud (Free)

1. **Push to GitHub**
```bash
git init
git add .
git commit -m "Initial commit: Decision Trees Dashboard"
git push origin main
```

2. **Deploy on Streamlit Cloud**
- Go to [share.streamlit.io](https://share.streamlit.io)
- Sign in with GitHub
- Select your repository
- Set main file path: `app.py`
- Deploy!

Your app will be live at: `https://[your-username]-decision-trees-dashboard.streamlit.app`

---

## 🛠️ Tech Stack

- **Frontend**: Streamlit 1.39+
- **Visualization**: Plotly 5.18, Matplotlib 3.8
- **ML Library**: scikit-learn 1.5+
- **Data Processing**: pandas 2.2, numpy 1.26
- **Deployment**: Streamlit Cloud

---

## 📝 Features Roadmap

- [ ] Support for regression trees (continuous targets)
- [ ] Ensemble methods (Random Forest, Gradient Boosting)
- [ ] Cross-validation and hyperparameter optimization
- [ ] Cost-complexity pruning visualization
- [ ] Export tree to GraphViz DOT format
- [ ] Batch processing for multiple datasets
- [ ] Rule comparison across different configurations

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

**Álvaro Andrés De Lamo**
- GitHub: [@aadl6-ua](https://github.com/aadl6-ua)
- LinkedIn: [Álvaro Andrés De Lamo](https://www.linkedin.com/in/alvaro-andres-de-lamo-50149b358)
- Portfolio: [aadl6-ua.github.io](https://aadl6-ua.github.io)

---

## 🎓 Academic Context

This project was developed as part of the **Decision and Association Rules in Knowledge Data** course at the **University of Silesia** (Poland) during an Erasmus+ exchange program (October 2024 - February 2025) as part of the Computer Engineering degree at Universidad de Alicante, Spain.

---

## 🙏 Acknowledgments

- University of Silesia for the Decision and Association Rules course
- Streamlit team for the amazing framework
- scikit-learn contributors for robust ML algorithms

---

**⭐ If you find this project useful, please consider giving it a star!**
