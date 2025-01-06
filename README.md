
# Credit Card Fraud Detection

This project leverages machine learning and deep learning models to detect fraudulent transactions from credit card data. The dataset includes anonymized features to protect user privacy and labels indicating whether a transaction is fraudulent.

## Dataset Description
The dataset is sourced from [Hugging Face](https://huggingface.co/datasets/tanzuhuggingface/creditcardfraudtraining) and includes the following features:
- **time_elapsed**: The time elapsed since the first transaction in the dataset.
- **amt**: The transaction amount.
- **lat** and **long**: The latitude and longitude of the transaction.
- **cc_num**: Credit Card number.
- **is_fraud**: The target label indicating whether the transaction is fraudulent (1) or non-fraudulent (0).

### Data Characteristics
- **Balanced Dataset**: The dataset has been balanced to ensure equal representation of fraudulent and non-fraudulent transactions.
- **Preprocessed Features**: All features are anonymized to maintain privacy.

## Features
- Exploratory Data Analysis (EDA) with visualizations for understanding data patterns.
- Implementation of machine learning models including:
  - Logistic Regression
  - Gradient Boosting Classifier
  - Random Forest Classifier
  - Decision Tree Classifier
  - XGBoost Classifier
  - KNN
- Evaluation of deep learning models such as:
  - Long Short-Term Memory (LSTM)
  - Gated Recurrent Unit (GRU)
- Model evaluation using metrics like accuracy, precision, recall, F1-score, and ROC-AUC score.
- Confusion matrices and ROC curves for visual performance analysis.

## Installation
1. Clone the repository.
   ```bash
   git clone https://github.com/SabaIsrar/Project
   ```

## Usage
1. Open the notebook CodeFile_Final.ipynb in Jupyter Notebook or Google Colab.
2. Follow the sections:
   - **EDA**: Understand the data distribution.
   - **PCA**: Analyze feature importance.
   - **Machine Learning Models**: Train and evaluate various ML models.
   - **Deep Learning Models**: Train and evaluate LSTM and GRU models.
3. Adjust hyperparameters as needed in the respective sections.
4. Run the cells to train and evaluate the models.

## Results
The XGBoost  showed the best performance with the following metrics:
- **Accuracy**: 99%
- **Precision**: 99%
- **Recall**: 99%
- **F1-Score**: 99%

## Visualizations
- Feature distributions and correlations using `Seaborn` and `Matplotlib`.
- Confusion matrices and ROC curves for model evaluation.

## Notes
- Ensure you have adequate computational resources for training deep learning models, especially LSTM and GRU.
- Use Google Colab for free GPU resources if needed.

## Contributing
Feel free to fork the repository, submit issues, or make pull requests for improvements.
