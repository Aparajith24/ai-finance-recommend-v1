import streamlit as st
import pandas as pd
from test import rf_trust, gb_outcome, clustering_features, kmeans, scaler, X

def predict_trust_and_recommendations(user_data):

    rec_type_mapping = {"Low Risk": 0, "Balanced": 1, "High Risk": 2}
    risk_level_mapping = {"Low": 0, "Medium": 1, "High": 2}

    user_data["rec_type"] = rec_type_mapping[user_data["rec_type"]]
    user_data["risk_level"] = risk_level_mapping[user_data["risk_level"]]

    user_df_for_prediction = pd.DataFrame(columns=X.columns)
    user_df_for_prediction.loc[0] = 0
    
    for column in X.columns:
        if column in user_data:
            user_df_for_prediction.loc[0, column] = user_data[column]
    
    if 'trust_efficiency_ratio' in X.columns:
        user_df_for_prediction['trust_efficiency_ratio'] = user_data['trust_score'] / (user_data['time_reviewing_recs'] + 1)
    
    if 'recommendation_acceptance' in X.columns:
        user_df_for_prediction['recommendation_acceptance'] = 1 - user_data['override_frequency']
    
    predicted_trust = rf_trust.predict(user_df_for_prediction)[0]
    predicted_outcome = gb_outcome.predict(user_df_for_prediction)[0]
    
    user_df_full = pd.DataFrame([user_data])
    missing_features = [feat for feat in clustering_features if feat not in user_df_full.columns]
    for feat in missing_features:
        user_df_full[feat] = 0
    
    user_features = user_df_full[clustering_features].copy()
    user_scaled = scaler.transform(user_features)
    user_cluster = kmeans.predict(user_scaled)[0]
    
    if user_data['risk_tolerance'] > 5:
        rec_strategy = "Aggressive growth strategy with high-risk investments"
        investment_recommendations = [
            "Large-cap growth stocks: Reliance Industries, HDFC Bank, Tata Consultancy Services (TCS)",
            "Mid & small-cap stocks: Deepak Nitrite, Balkrishna Industries, Dixon Technologies",
            "Thematic/sectoral funds: ICICI Prudential Technology Fund, Mirae Asset Healthcare Fund",
            "Cryptocurrency exposure (via ETFs if available in India)",
        ]
    elif user_data['risk_tolerance'] > 3:
        rec_strategy = "Balanced approach with a mix of growth and stability"
        investment_recommendations = [
            "Blue-chip stocks: Infosys, HUL, Kotak Mahindra Bank",
            "Index funds: NIFTY 50 ETF, Sensex ETF",
            "Hybrid mutual funds: HDFC Balanced Advantage Fund, SBI Equity Hybrid Fund",
            "Corporate bonds: Tata Capital NCDs, Mahindra Finance Bonds",
        ]
    else:
        rec_strategy = "Conservative strategy with a focus on stability"
        investment_recommendations = [
            "Fixed deposits: SBI FD, HDFC FD with 5+ years lock-in",
            "Government-backed bonds: RBI Floating Rate Bonds, GOI Savings Bonds",
            "Gold ETFs: Nippon India Gold ETF, HDFC Gold ETF",
            "Stable dividend stocks: Power Grid Corporation, NTPC, Coal India",
        ]
    
    if predicted_trust < 30:
        transparency_level = "High transparency with detailed explanations for all recommendations"
    else:
        transparency_level = "Standard transparency with option to view detailed explanations"
    
    return {
        'predicted_trust': predicted_trust,
        'predicted_financial_outcome': predicted_outcome,
        'user_segment': user_cluster,
        'recommended_strategy': rec_strategy,
        'transparency_approach': transparency_level,
        'investment_recommendations': investment_recommendations
    }

st.title("AI-Assisted Financial Trust & Efficiency Analyzer (India)")
st.header("Enter Your Details")
trust_score = st.slider("Trust Score", 0, 100, 50)
financial_literacy = st.slider("Financial Literacy", 0, 10, 5)
risk_tolerance = st.slider("Risk Tolerance", 0, 10, 5)
follow_through_rate = st.slider("Follow-Through Rate", 0.0, 1.0, 0.5)
override_frequency = st.slider("Override Frequency", 0.0, 1.0, 0.5)
app_usage_frequency = st.slider("App Usage Frequency", 0, 10, 5)
time_reviewing_recs = st.slider("Time Reviewing Recommendations (minutes)", 0, 60, 15)
perceived_usefulness = st.slider("Perceived Usefulness", 0, 5, 3)
perceived_accuracy = st.slider("Perceived Accuracy", 0, 5, 3)
explanation_quality = st.slider("Explanation Quality", 0, 5, 3)
transparency_rating = st.slider("Transparency Rating", 0, 5, 3)
privacy_concerns = st.slider("Privacy Concerns", 0, 5, 2)
rec_type = st.selectbox("Recommendation Type", options=["Balanced", "High Risk", "Low Risk"], index=0)
risk_level = st.selectbox("Risk Level", options=["Low", "Medium", "High"], index=1)

if st.button("Get Results"):
    user_input = {
        'trust_score': trust_score,
        'financial_literacy': financial_literacy,
        'risk_tolerance': risk_tolerance,
        'follow_through_rate': follow_through_rate,
        'override_frequency': override_frequency,
        'app_usage_frequency': app_usage_frequency,
        'time_reviewing_recs': time_reviewing_recs,
        'perceived_usefulness': perceived_usefulness,
        'perceived_accuracy': perceived_accuracy,
        'explanation_quality': explanation_quality,
        'transparency_rating': transparency_rating,
        'privacy_concerns': privacy_concerns,
        'rec_type': rec_type,
        'risk_level': risk_level
    }
    prediction_result = predict_trust_and_recommendations(user_input)
    
    st.success(f"Predicted Trust Score: {prediction_result['predicted_trust']}")
    st.success(f"Predicted Financial Outcome: {prediction_result['predicted_financial_outcome']}")
    st.success(f"User Segment: {prediction_result['user_segment']}")
    st.success(f"Recommended Strategy: {prediction_result['recommended_strategy']}")
    st.success(f"Transparency Approach: {prediction_result['transparency_approach']}")

    st.subheader("Potential Investment Recommendations (India)")
    for investment in prediction_result['investment_recommendations']:
        st.write(f"- {investment}")

st.write("This tool provides AI-driven financial analysis and investment recommendations for the Indian market.")
