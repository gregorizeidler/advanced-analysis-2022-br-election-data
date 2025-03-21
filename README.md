# Advanced Analysis of 2022 Election Data

This project consists of an interactive dashboard created with Streamlit for advanced analysis of Brazilian election data from 2022, focusing on the identification of atypical statistical patterns and sophisticated statistical analyses.

## Video Demonstration

A video demonstration of this project is available here:
[Watch Demo Video](https://drive.google.com/file/d/1cfEhRCeYBOLD_QGAARmuhVEIGkWQyzmG/view?usp=sharing)

## Main Features

- **Modern Interface**: Dashboard organized in tabs for easy navigation
- **Multivariate Analysis**: Identification of atypical statistical patterns considering multiple variables
- **Advanced Visualizations**: Interactive charts with Plotly, heat maps, and various types of visualizations
- **Statistical Methods**: Hypothesis testing, distribution analysis, and Benford's Law
- **Machine Learning**: Identification of outliers with algorithms such as Isolation Forest, DBSCAN, GMM, and Autoencoder
- **Export**: Export of data and reports in various formats (CSV, Excel, JSON, Parquet)

## Requirements

To run this project, you need to have Python 3.7+ installed, along with the following libraries:

```bash
pip install -r requirements.txt
```

The main dependencies include:
- streamlit
- pandas
- matplotlib
- numpy
- scikit-learn
- plotly
- seaborn
- statsmodels
- xgboost
- shap
- tensorflow (optional, for autoencoder)
- geopandas (optional, for geographic visualizations)

## Required Data

The application needs the original 2022 election data file:

- `votacao_secao_2022_BR.csv`: file with voting data by electoral section in Brazil in 2022

You can download this file from the [Superior Electoral Court (TSE) website](https://dadosabertos.tse.jus.br/dataset/resultados-2022), in the open data section.

## How to Run

1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Place the `votacao_secao_2022_BR.csv` file in the project's root folder
4. Run the Streamlit application: `streamlit run app.py`

## Methods for Identifying Atypical Patterns

The dashboard implements various statistical methods for identifying atypical patterns:

1. **IQR (Interquartile Range)**: Identifies outliers based on data dispersion
2. **Z-Score**: Detects atypical values based on standard deviation
3. **Isolation Forest**: Unsupervised algorithm that isolates atypical observations
4. **DBSCAN**: Density-based clustering algorithm for outlier detection
5. **GMM (Gaussian Mixture Model)**: Identifies atypical patterns based on probabilistic models
6. **PCA**: Principal component analysis with Mahalanobis distance
7. **Autoencoder**: Neural network for identifying uncommon patterns (requires TensorFlow)
8. **Ensemble**: Combination of the previous methods for more robust identification

## Statistical Analyses

The dashboard includes the following statistical analyses:

- **Descriptive Analysis**: Basic statistics, histograms, and distribution visualizations
- **Normality Tests**: Shapiro-Wilk and D'Agostino-Pearson
- **Benford's Law Analysis**: Verification of compliance with Benford's Law for identification of unusual statistical patterns
- **Correlation Analysis**: Identification of relationships between variables
- **Regression**: Statistical models to explain relationships
- **Temporal Analysis**: Visualization of data over time

## Explainability with Machine Learning

The dashboard also offers:

- **Feature Importance**: Visualization of the most relevant variables for identifying atypical patterns
- **SHAP Values**: Explanation of the contribution of each feature in the analyses
- **Explainable Models**: Use of interpretable models such as decision trees

## Export and Reports

Export functionalities:

- Export of raw data in various formats (CSV, Excel, JSON, Parquet)
- Complete statistical reports
- Export of list of identified atypical patterns
- Charts and visualizations

## Notes

On first execution, the application will convert the CSV file to Parquet format, which may take a few minutes due to the large volume of data. In subsequent executions, the Parquet file will be used directly, resulting in much faster loading.

## Purpose and Neutrality

This project has a purely scientific and statistical purpose, focused on data analysis and identification of statistical patterns. It does not have any political connotation or intention to question the legitimacy or integrity of the electoral process.

The atypical patterns identified are natural statistical phenomena present in large data sets, and their presence does not suggest irregularities in the process. This work falls within the field of data science and statistical analysis, contributing to the understanding of distributions and behaviors in electoral data.

## Contributions

Contributions are welcome! Feel free to open issues or pull requests with improvements.
