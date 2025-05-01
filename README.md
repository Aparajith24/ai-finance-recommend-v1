# AI-Assisted Personal Finance Tools: Analyzing User Trust and Efficiency

This project analyzes how users interact with AI-powered financial recommendation systems, focusing specifically on trust factors and efficiency metrics that lead to improved financial outcomes.

## Overview

This analysis examines the relationships between user characteristics, trust in AI systems, efficiency of interactions, and financial outcomes. The project employs data exploration, statistical analysis, clustering, and machine learning to derive actionable insights about how to build better AI financial advisory tools.

## Features

- **Data Exploration & Preprocessing**: Comprehensive analysis of user behavioral and financial data
- **Feature Engineering**: Creation of composite metrics like trust-efficiency ratio and financial outcome measures
- **Exploratory Analysis**: Visualization of key relationships and distributions
- **Trust Analysis**: Identification of key factors that influence user trust in AI financial recommendations
- **Efficiency Analysis**: Examination of how efficiency metrics correlate with financial outcomes
- **User Segmentation**: Clustering analysis to identify distinct user groups
- **Predictive Modeling**: Machine learning models to predict trust scores and financial outcomes
- **Ethics & Privacy Analysis**: Investigation of how privacy concerns impact system usage and trust
- **Recommendation System Evaluation**: Assessment of different recommendation approaches

## Key Findings

1. **Trust Factors**: The analysis identifies the most influential factors affecting user trust in AI financial recommendations
2. **User Segments**: The project identifies distinct user segments with different behavioral patterns and trust levels
3. **Recommendation Effectiveness**: Analysis reveals which recommendation types lead to better financial outcomes
4. **Privacy Impact**: The research quantifies how privacy concerns affect user trust and system engagement

## Prerequisites

- Python 3.7+
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

## Installation

```bash
# Clone the repository
git clone https://github.com/Aparajith24/ai-finance-recommend-v1.git

# Install required packages
pip install -r requirements.txt
```

## Usage

```python
# Run the main analysis script
python main.py
```

```python
# Run the web frontend streamlit app
python streamlit.py
```

## Output Files

The analysis generates several visualization files:
- `trust_score_distribution.png`: Distribution of user trust scores
- `correlation_matrix.png`: Correlation between key variables
- `trust_factors.png`: Analysis of factors affecting trust
- `trust_by_age.png`: Trust scores across different age groups
- `efficiency_correlation.png`: Correlation between efficiency metrics and outcomes
- `follow_through_outcomes.png`: Relationship between follow-through rate and financial results
- `elbow_method.png`: Analysis for optimal number of clusters
- `user_segments.png`: Visualization of user segments
- `trust_feature_importance.png`: Most important features for predicting trust
- `outcome_feature_importance.png`: Features that best predict financial outcomes
- `privacy_impact.png`: Impact of privacy concerns on trust and usage
- `transparency_trust.png`: Relationship between transparency and trust
- `recommendation_type_analysis.png`: Performance analysis by recommendation type

## Dataset

The analysis uses `financial_ai_dataset.csv` which contains the following key variables:
- User demographics (age, education level, income)
- Financial metrics (savings rate, investment returns, debt reduction)
- Behavioral metrics (trust score, override frequency, follow-through rate)
- System characteristics (transparency rating, explanation quality)

## License

[MIT License](LICENSE)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Contact

Project Link: [https://github.com/Aparajith24/ai-finance-recommend-v1.git](https://github.com/Aparajith24/ai-finance-recommend-v1.git)
