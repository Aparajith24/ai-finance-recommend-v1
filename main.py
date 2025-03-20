# AI-Assisted Personal Finance Tools: Analyzing User Trust and Efficiency
# ===================================================================

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings('ignore')

# Set the styling for visualizations
plt.style.use('ggplot')
sns.set_palette("Set2")
sns.set_context("notebook", font_scale=1.2)

# Load the dataset
df = pd.read_csv('data/financial_ai_dataset.csv')

# 1. Data Exploration and Preprocessing
# =====================================

# Display basic information about the dataset
print("Dataset Shape:", df.shape)
print("\nDataset Columns:", df.columns.tolist())
print("\nSample Data:")
print(df.head())

# Check for missing values
print("\nMissing Values:", df.isnull().sum().sum())

# Basic statistical summary
print("\nBasic Statistics:")
print(df.describe())

# Convert categorical data if needed
categorical_cols = ['education_level', 'income_level', 'rec_type', 'risk_level']
for col in categorical_cols:
    if col in df.columns and df[col].dtype == 'object':
        df[col] = LabelEncoder().fit_transform(df[col])

# 2. Feature Engineering and Analysis
# ==================================

# Create new composite features
df['trust_efficiency_ratio'] = df['trust_score'] / (df['time_reviewing_recs'] + 1)  # Adding 1 to avoid division by zero
df['financial_outcome_composite'] = (df['savings_rate_change'] + 
                                     df['investment_returns']/20 + 
                                     df['debt_reduction'] + 
                                     df['net_worth_growth']) / 4

df['recommendation_acceptance'] = 1 - df['override_frequency']
df['trust_perception_gap'] = df['trust_score'] - ((df['perceived_usefulness'] + df['perceived_accuracy']) * 10)

# 3. Exploratory Data Analysis
# ===========================

# Setup EDA notebook section with visualizations
plt.figure(figsize=(10, 6))
sns.histplot(df['trust_score'], kde=True)
plt.title('Distribution of Trust Scores')
plt.xlabel('Trust Score')
plt.savefig('trust_score_distribution.png')
plt.close()

plt.figure(figsize=(12, 8))
correlation = df.corr()
mask = np.triu(correlation)
sns.heatmap(correlation, annot=False, mask=mask, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix')
plt.tight_layout()
plt.savefig('correlation_matrix.png')
plt.close()

# 4. Trust Analysis
# ================

# Examine factors affecting trust
trust_factors = ['financial_literacy', 'app_usage_frequency', 'explanation_quality', 
                 'transparency_rating', 'privacy_concerns', 'override_frequency']

plt.figure(figsize=(14, 10))
for i, factor in enumerate(trust_factors, 1):
    plt.subplot(3, 2, i)
    sns.scatterplot(x=df[factor], y=df['trust_score'], alpha=0.7)
    plt.title(f'Trust Score vs {factor}')
    plt.tight_layout()
plt.savefig('trust_factors.png')
plt.close()

# Group by age ranges and calculate mean trust scores
df['age_group'] = pd.cut(df['age'], bins=[18, 30, 45, 60, 100], labels=['18-30', '31-45', '46-60', '60+'])
trust_by_age = df.groupby('age_group')['trust_score'].mean().reset_index()

plt.figure(figsize=(10, 6))
sns.barplot(x='age_group', y='trust_score', data=trust_by_age)
plt.title('Average Trust Score by Age Group')
plt.ylabel('Average Trust Score')
plt.xlabel('Age Group')
plt.savefig('trust_by_age.png')
plt.close()

# 5. Efficiency Analysis
# ====================

# Analyze relationship between efficiency metrics and financial outcomes
efficiency_metrics = ['time_reviewing_recs', 'follow_through_rate', 'override_frequency']
financial_outcomes = ['savings_rate_change', 'investment_returns', 'budget_adherence', 'debt_reduction', 'net_worth_growth']

# Efficiency correlation matrix
efficiency_corr = df[efficiency_metrics + financial_outcomes].corr()
plt.figure(figsize=(12, 8))
sns.heatmap(efficiency_corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation between Efficiency Metrics and Financial Outcomes')
plt.tight_layout()
plt.savefig('efficiency_correlation.png')
plt.close()

# Scatter plot of follow-through rate vs financial outcomes
plt.figure(figsize=(12, 8))
for i, outcome in enumerate(financial_outcomes, 1):
    plt.subplot(2, 3, i)
    sns.scatterplot(x='follow_through_rate', y=outcome, data=df, alpha=0.7, hue='risk_tolerance')
    plt.title(f'Follow-through Rate vs {outcome}')
plt.tight_layout()
plt.savefig('follow_through_outcomes.png')
plt.close()

# 6. User Segmentation
# ===================

# Perform clustering to identify user segments based on trust and behavior
# Select features for clustering
clustering_features = ['trust_score', 'financial_literacy', 'risk_tolerance', 
                       'follow_through_rate', 'override_frequency', 'app_usage_frequency']

# Scale the data
X_cluster = df[clustering_features].copy()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

# Determine optimal number of clusters using the elbow method
inertia = []
k_range = range(1, 11)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(k_range, inertia, marker='o')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.savefig('elbow_method.png')
plt.close()

# Apply K-means with the chosen number of clusters (4 in this case)
n_clusters = 4
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
df['cluster'] = kmeans.fit_predict(X_scaled)

# Analyze cluster characteristics
cluster_profile = df.groupby('cluster')[clustering_features + ['financial_outcome_composite']].mean()
print("\nCluster Profiles:")
print(cluster_profile)

# Visualize clusters using PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(X_scaled)
df['pca1'] = pca_result[:, 0]
df['pca2'] = pca_result[:, 1]

plt.figure(figsize=(10, 8))
sns.scatterplot(x='pca1', y='pca2', hue='cluster', data=df, palette='viridis', s=100, alpha=0.7)
plt.title('User Segments Visualization')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.savefig('user_segments.png')
plt.close()

# 7. Predictive Modeling
# =====================

# Build models to predict trust scores and financial outcomes

# Prepare features and target variables
X = df.drop(['trust_score', 'financial_outcome_composite', 'pca1', 'pca2', 'cluster', 'age_group'], axis=1)
y_trust = df['trust_score']
y_outcome = df['financial_outcome_composite']

# Split data into training and testing sets
X_train, X_test, y_trust_train, y_trust_test = train_test_split(X, y_trust, test_size=0.2, random_state=42)
_, _, y_outcome_train, y_outcome_test = train_test_split(X, y_outcome, test_size=0.2, random_state=42)

# Trust Score prediction model
print("\nTraining Trust Score Prediction Model...")
rf_trust = RandomForestRegressor(n_estimators=100, random_state=42)
rf_trust.fit(X_train, y_trust_train)

# Financial outcome prediction model
print("Training Financial Outcome Prediction Model...")
gb_outcome = GradientBoostingRegressor(n_estimators=100, random_state=42)
gb_outcome.fit(X_train, y_outcome_train)

# Evaluate models
y_trust_pred = rf_trust.predict(X_test)
y_outcome_pred = gb_outcome.predict(X_test)

print("\nTrust Score Model Performance:")
print(f"RMSE: {np.sqrt(mean_squared_error(y_trust_test, y_trust_pred)):.4f}")
print(f"R²: {r2_score(y_trust_test, y_trust_pred):.4f}")
print("Accuracy:", rf_trust.score(X_test, y_trust_test))

print("\nFinancial Outcome Model Performance:")
print(f"RMSE: {np.sqrt(mean_squared_error(y_outcome_test, y_outcome_pred)):.4f}")
print(f"R²: {r2_score(y_outcome_test, y_outcome_pred):.4f}")
print("Accuracy:", gb_outcome.score(X_test, y_outcome_test))

# Feature importance for trust model
trust_importance = permutation_importance(rf_trust, X_test, y_trust_test, n_repeats=10, random_state=42)
trust_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': trust_importance.importances_mean
}).sort_values('Importance', ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=trust_importance_df.head(15))
plt.title('Feature Importance for Trust Score Prediction')
plt.tight_layout()
plt.savefig('trust_feature_importance.png')
plt.close()

# Feature importance for financial outcome model
outcome_importance = permutation_importance(gb_outcome, X_test, y_outcome_test, n_repeats=10, random_state=42)
outcome_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': outcome_importance.importances_mean
}).sort_values('Importance', ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=outcome_importance_df.head(15))
plt.title('Feature Importance for Financial Outcome Prediction')
plt.tight_layout()
plt.savefig('outcome_feature_importance.png')
plt.close()

# 8. Ethics and Privacy Analysis
# ============================

# Analyze how privacy concerns impact trust and usage
privacy_impact = df.groupby('privacy_concerns')[['trust_score', 'app_usage_frequency', 'follow_through_rate']].mean()
print("\nImpact of Privacy Concerns:")
print(privacy_impact)

plt.figure(figsize=(10, 6))
privacy_impact.plot(kind='bar')
plt.title('Impact of Privacy Concerns on Trust and Usage')
plt.xlabel('Privacy Concern Level')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('privacy_impact.png')
plt.close()

# Analyze relationship between transparency and trust
plt.figure(figsize=(10, 6))
sns.boxplot(x='transparency_rating', y='trust_score', data=df)
plt.title('Relationship Between Transparency and Trust')
plt.xlabel('Transparency Rating')
plt.ylabel('Trust Score')
plt.savefig('transparency_trust.png')
plt.close()

# 9. Recommendation System Evaluation
# =================================

# Analyze effectiveness of different recommendation types
rec_type_analysis = df.groupby('rec_type')[['follow_through_rate', 'financial_outcome_composite', 'user_satisfaction']].mean()
print("\nRecommendation Type Analysis:")
print(rec_type_analysis)

plt.figure(figsize=(12, 6))
rec_type_analysis.plot(kind='bar')
plt.title('Performance by Recommendation Type')
plt.xlabel('Recommendation Type')
plt.ylabel('Average Value')
plt.xticks(rotation=0)
plt.legend(loc='best')
plt.tight_layout()
plt.savefig('recommendation_type_analysis.png')
plt.close()

# 10. Insights and Recommendations
# =============================

# Prepare a summary of key findings
print("\n======================================")
print("Key Insights and Recommendations")
print("======================================")

# Trust factors analysis
print("\n1. Trust Factors Analysis:")
top_trust_factors = trust_importance_df.head(5)
print(f"Top factors influencing trust: {', '.join(top_trust_factors['Feature'].tolist())}")

# User segments analysis
print("\n2. User Segment Analysis:")
for cluster in range(n_clusters):
    segment_size = (df['cluster'] == cluster).sum()
    segment_pct = segment_size / len(df) * 100
    avg_trust = df[df['cluster'] == cluster]['trust_score'].mean()
    avg_outcome = df[df['cluster'] == cluster]['financial_outcome_composite'].mean()
    print(f"Segment {cluster+1} ({segment_size} users, {segment_pct:.1f}%): Avg Trust = {avg_trust:.1f}, Avg Financial Outcome = {avg_outcome:.3f}")

# Recommendation effectiveness
print("\n3. Recommendation Effectiveness:")
best_rec = rec_type_analysis['financial_outcome_composite'].idxmax()
print(f"Most effective recommendation type: {best_rec}")

# Privacy and transparency
print("\n4. Privacy and Transparency:")
trust_diff = privacy_impact.loc[1]['trust_score'] - privacy_impact.loc[5]['trust_score']
print(f"Trust score difference between lowest and highest privacy concerns: {trust_diff:.2f}")

