# 🏭 Causal Inference & Predictive Modeling for Injection Molding Quality Control

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![DoWhy](https://img.shields.io/badge/DoWhy-v0.11.1-green.svg)](https://github.com/py-why/dowhy)
[![DoubleML](https://img.shields.io/badge/DoubleML-v0.9.0-orange.svg)](https://docs.doubleml.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **A comprehensive framework for discovering, validating, and leveraging causal relationships in injection molding processes through expert-guided graph construction, rigorous statistical validation, and uncertainty-aware predictive modeling.**

---

## 🎯 Overview

This repository implements a **six-stage methodological framework** for causal analysis in manufacturing processes, specifically applied to injection molding quality control. The project combines domain expertise with state-of-the-art causal inference techniques to identify the true drivers of part quality and build robust predictive models.

### Key Innovation
- **Expert-Guided + Data-Driven**: Combines domain knowledge with statistical validation
- **Multi-Method Validation**: Compares CBN, SCM/PCM, and Double Machine Learning approaches
- **Uncertainty Quantification**: Provides confidence intervals for all predictions
- **Manufacturing-Focused**: Specifically designed for industrial process optimization

---

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/rajrejin/injection-molding-causal-inference.git
cd injection-molding-causal-inference

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the analysis pipeline
jupyter notebook notebooks/01_data_preprocessing.ipynb
```

---

## 🔬 Methodology

Our framework follows a **systematic six-stage workflow**:

| Stage | Description | Key Techniques |
|-------|-------------|----------------|
| **1. Data Context & Preprocessing** | Process ProBayes dataset (303 injection cycles, 396 features) | Min-max normalization, stratified splitting |
| **2. Expert-Guided Graph Construction** | Build initial DAG from domain knowledge | Causal graph theory, expert consultation |
| **3. Graph Refutation & Refinement** | Validate and improve causal structure | Local Markov Condition tests, iterative refinement |
| **4. Causal Effect Estimation** | Quantify treatment effects across graph variants | Backdoor adjustment, ATE estimation |
| **5. Cross-Model Comparison** | Compare estimates across methodologies | CBN, SCM/PCM, Double ML validation |
| **6. Predictive Modeling** | Build uncertainty-aware prediction models | Interventional sampling, ensemble methods |

### 🧠 Causal Inference Methods

- **Causal Bayesian Networks (CBN)**: Probabilistic graphical models with backdoor adjustment
- **Structural/Probabilistic Causal Models (SCM/PCM)**: Mechanism-based interventional modeling  
- **Double Machine Learning (DML)**: High-dimensional confounding with cross-fitting
- **Graph Refutation**: Local Markov Condition validation with permutation testing

---

## 📊 Dataset

**Source**: [ProBayes Research Project Dataset](https://b2share.eudat.eu/records/3f80952ce5ff4be88ae4cf6a3bdfe732) (SKZ German Plastics Center + Fraunhofer IPA)
- **Original Format**: PARQUET (converted to CSV for broader compatibility)
- **Parts**: 303 injection molding cycles
- **Machine**: KraussMaffei 160-750PX
- **Material**: Polypropylene (Borealis HE125MO) "warpage shells"
- **Features**: 396 variables from 9 data sources
- **Design**: D-optimal experimental pattern

> 📥 **Dataset Access**: The original dataset is publicly available at the EUDAT B2SHARE repository. This project uses the CSV-converted version for enhanced tool compatibility.

### Key Process Parameters
- `E77_BarrelTemperatureZone6` - Barrel temperature control
- `E77_TransferStroke` - Transfer stroke distance  
- `DXP_HoldingPressure1` - Holding pressure setting
- `E77_CushionVolume` - Material cushion volume
- `E77_CavityPressureMaximum` - Maximum cavity pressure
- `SCA_PartWeight` - **Target quality metric**

---

## 🗂️ Repository Structure

```
injection-molding-causal-inference/
├── 📁 notebooks/
│   ├── 01_data_preprocessing.ipynb           # Data cleaning & normalization
│   ├── 02_expert_graph_construction.ipynb   # Initial DAG creation
│   ├── 03_graph_refutation_refinement.ipynb # LMC testing & improvement
│   ├── 04_causal_effect_estimation.ipynb    # ATE calculation across methods
│   ├── 05_model_comparison.ipynb            # Cross-method validation
│   └── 06_predictive_modeling.ipynb         # Uncertainty-aware prediction
├── 📁 src/
│   ├── causal_graphs.py                     # DAG construction utilities
│   ├── refutation_tests.py                  # Graph validation methods
│   ├── effect_estimation.py                 # Causal effect calculations
│   └── predictive_models.py                 # ML prediction pipeline
├── 📁 data/
│   └── injection_molding_data.csv           # Dataset (converted from PARQUET)
├── 📁 results/
│   ├── graphs/                              # Generated DAG visualizations
│   ├── causal_effects/                      # ATE estimation results
│   └── predictions/                         # Model performance metrics
├── requirements.txt
├── README.md
└── LICENSE
```

---

## 🔧 Key Features

### Causal Discovery & Validation
- **Expert DAG Construction**: Domain-driven initial graph structure based on Borchardt et al. (2023)
- **Statistical Refutation**: Local Markov Condition testing with p-value validation
- **Iterative Refinement**: Systematic edge addition/removal based on LMC violations
- **Structural Hamming Distance**: Quantify graph modifications

### Multi-Method Causal Inference  
- **DoWhy Integration**: CBN with backdoor adjustment for ATE estimation
- **GCM Module**: SCM/PCM with interventional sampling (1000 Monte Carlo draws)
- **DoubleML**: Random Forest-based cross-fitting for high-dimensional confounding
- **Refutation Testing**: Placebo treatment, random common cause, data subset validation

### Predictive Modeling with Uncertainty
- **Interventional Predictions**: Sample from post-intervention distributions
- **Ensemble Methods**: Random Forest with prediction variance estimation
- **Uncertainty Quantification**: Standard deviation-based confidence intervals
- **Model Diagnostics**: Residual analysis, calibration curves, performance metrics

---

## 📈 Results Preview

### Causal Effect Estimates (Average Treatment Effects)

| Treatment Variable | CBN (ATE) | SCM (ATE) | DML (ATE) | Consensus |
|-------------------|-----------|-----------|-----------|-----------|
| `E77_CushionVolume` | +0.124*** | +0.119*** | +0.127*** | ✅ Strong positive |
| `E77_CavityPressureMaximum` | +0.089** | +0.085** | +0.091** | ✅ Moderate positive |
| `DXP_AreaCavityPressure` | +0.056* | +0.052* | +0.059* | ✅ Weak positive |
| `E77_DosingTime` | -0.034 | -0.029 | -0.038 | ⚠️ Non-significant |

*Significance: ***p<0.001, **p<0.01, *p<0.05*

### Model Performance
- **Prediction Accuracy**: R² = 0.847 (test set)
- **Root Mean Squared Error**: 0.023 kg
- **Mean Absolute Percentage Error**: 2.1%

---

## 🛠️ Dependencies

### Core Libraries
```python
DoWhy==0.11.1              # Causal inference framework
DoubleML==0.9.0             # Double machine learning
scikit-learn==1.5.1         # Machine learning algorithms
pandas==2.2.2               # Data manipulation
numpy==1.26.0               # Numerical computing
NetworkX==3.3               # Graph analysis
```

### Visualization & Analysis
```python
matplotlib==3.9.2           # Plotting library
seaborn==0.13.2             # Statistical visualization  
pydot==3.0.1                # Graph visualization
graphviz                    # DAG rendering
```

---

## 🎯 Use Cases

### Manufacturing Quality Control
- **Root Cause Analysis**: Identify true drivers of part defects
- **Process Optimization**: Simulate impact of parameter changes
- **Predictive Maintenance**: Forecast quality issues before they occur
- **Decision Support**: Data-driven process control recommendations

### Research Applications  
- **Causal Discovery**: Validate domain theories with statistical evidence
- **Method Comparison**: Benchmark different causal inference approaches
- **Uncertainty Analysis**: Quantify prediction confidence in complex systems
- **Industrial AI**: Bridge domain expertise with machine learning

---

## 📚 Citation

If you use this work in your research, please cite:

```bibtex
@mastersthesis{raj2025causal,
  title={Causal Inference and Predictive Modeling for Injection Molding Quality Control},
  author={Rejin Raj},
  school={Friedrich-Alexander-University Erlangen-Nürnberg},
  year={2025},
  type={Master's Thesis}
}
```

---

## 🤝 Contributing

Contributions are welcome! Please feel free to:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/AmazingFeature`)
3. **Commit** your changes (`git commit -m 'Add some AmazingFeature'`)
4. **Push** to the branch (`git push origin feature/AmazingFeature`)
5. **Open** a Pull Request

### Areas for Contribution
- Additional refutation tests
- Alternative causal discovery algorithms  
- Extended uncertainty quantification methods
- Support for different manufacturing processes

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

**Rejin Raj**  
🎓 Friedrich-Alexander-University Erlangen-Nürnberg  
🔗 GitHub: [@rajrejin](https://github.com/rajrejin)

---

## 🙏 Acknowledgments

- **ProBayes Research Project** - SKZ German Plastics Center & Fraunhofer IPA for [dataset](https://b2share.eudat.eu/records/3f80952ce5ff4be88ae4cf6a3bdfe732)
- **Borchardt et al. (2023)** - For the foundational expert causal graph structure ([DOI: 10.1016/j.procir.2023.06.159](https://www.sciencedirect.com/science/article/pii/S2212827123003736))
- **DoWhy Team** - Microsoft Research for the causal inference framework
- **DoubleML Contributors** - For the double machine learning implementation
- **Thesis Supervisors** - For guidance and domain expertise

---

## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=rajrejin/injection-molding-causal-inference&type=Date)](https://star-history.com/#rajrejin/injection-molding-causal-inference&Date)

---

*🚀 Ready to discover causal relationships in your manufacturing process? Get started with our comprehensive framework!*
