# Imputation of Missing Data App

A Streamlit application for missing data imputation. Analyze, visualize, and handle missing values in datasets using various imputation techniques, demonstrated with a water potability dataset.

## TECH

- **Streamlit** - Web application framework
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **Scikit-learn** - Machine learning algorithms for imputation
- **Matplotlib & Seaborn** - Data visualization
- **MICE Forest** - Advanced imputation techniques

## FEATURES

### Data Analysis

- Upload and analyze CSV datasets
- Exploratory data analysis with summary statistics
- Missing value detection and visualization

### Missing Value Configuration

- Interactive missing value pattern configuration
- Custom missing value indicators
- Data preprocessing capabilities

### Imputation Methods

- **Simple Imputation**: Mean, Median, Mode
- **KNN Imputation**: K-Nearest Neighbors with configurable parameters
- **MICE (Multiple Imputation by Chained Equations)**: Advanced iterative imputation

### Results Comparison

- Side-by-side comparison of different imputation methods
- Performance metrics and visualizations
- Export functionality for imputed datasets

## INSTALLATION

1. Clone the repository:

```bash
git clone https://github.com/johnOfGod33/imputation-of-missing-data-app.git
cd imputation-of-missing-data-app
```

2. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run the application:

```bash
streamlit run src/main.py
```

The app will open in your browser at `http://localhost:8501`
