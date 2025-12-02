# ML Agent SDK - Generic Machine Learning Agent

A generic autonomous ML agent using the Claude Agent SDK that supports multiple datasets and ML pipelines.

## Overview

This project demonstrates a flexible ML agent that can autonomously:
- Load and explore different types of datasets (tabular, images)
- Preprocess data and engineer features
- Train and compare multiple ML models
- Generate predictions

The agent autonomously decides which tools to use and in what order based on the dataset type.

## Supported Datasets

### 1. Titanic (Tabular Data)
- **Task**: Survival prediction
- **Data**: CSV format with passenger information
- **Models**: Random Forest, Logistic Regression, Gradient Boosting

### 2. MNIST (Image Data)
- **Task**: Handwritten digit classification (0-9)
- **Data**: IDX format with 28x28 grayscale images
- **Models**: Random Forest, Logistic Regression, MLP, Gradient Boosting

## Installation

Install dependencies:
```bash
pip install -r requirements.txt
```

Set up your Anthropic API key:
```bash
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY
```

## Usage

### Run Generic Agent

**Titanic Dataset:**
```bash
python generic_agent.py titanic
```

**MNIST Dataset:**
```bash
python generic_agent.py mnist
```

### Run Dataset-Specific Agents

**Titanic only:**
```bash
python titanic_agent_sdk_2.py
```

## Project Structure

```
├── generic_agent.py          # Main generic agent (supports both datasets)
├── titanic_tools.py           # Titanic-specific ML operations
├── mnist_tools.py             # MNIST-specific ML operations
├── titanic_agent_sdk_2.py     # Titanic-only agent
├── data/
│   ├── titanic/
│   │   ├── train.csv
│   │   └── test.csv
│   └── mnist/
│       ├── train-images.idx3-ubyte
│       ├── train-labels.idx1-ubyte
│       ├── t10k-images.idx3-ubyte
│       └── t10k-labels.idx1-ubyte
└── requirements.txt
```

## How It Works

The agent uses an agentic loop where Claude autonomously:
1. Analyzes the current state
2. Selects appropriate tools
3. Executes them with optimal parameters
4. Interprets results
5. Continues until the task is complete

**Example Titanic execution:**
```
1. load_data → 891 training samples loaded
2. explore → Statistics and correlations analyzed
3. preprocess → Features engineered (family size, titles, encoding)
4. train → 3 models compared (RF, LR, GB)
5. predict → 418 predictions saved
```

**Example MNIST execution:**
```
1. load_mnist → 60,000 training images loaded
2. explore_mnist → Image statistics analyzed
3. preprocess_mnist → Normalized and flattened to 784 features
4. train_mnist → 3 models compared (RF, LR, MLP)
5. predict_mnist → 10,000 predictions saved
```

## Output

- **Titanic**: `predictions_titanic.csv` - Survival predictions
- **MNIST**: `predictions_mnist.txt` - Digit predictions

## Learn More

- [Claude Agent SDK Documentation](https://docs.anthropic.com/en/docs/build-with-claude/agent-sdk)
- [Titanic Competition (Kaggle)](https://www.kaggle.com/c/titanic)
- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)
