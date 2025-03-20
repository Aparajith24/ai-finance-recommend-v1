import streamlit as st
import pandas as pd
from main import rf_trust, gb_outcome, clustering_features, kmeans, scaler, X

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

# Helper function to convert text choices to numerical values
def convert_to_numerical_value(option, scale_type):
    if scale_type == "trust_score":
        options = {
            "Highly skeptical": 10,
            "Somewhat skeptical": 30,
            "Neutral": 50,
            "Somewhat trusting": 70,
            "Highly trusting": 90
        }
    elif scale_type == "financial_literacy":
        options = {
            "Beginner (new to investing)": 2,
            "Basic knowledge": 4,
            "Intermediate": 6,
            "Advanced": 8,
            "Expert (finance professional)": 10
        }
    elif scale_type == "risk_tolerance":
        options = {
            "Very conservative (avoid losses at all costs)": 1,
            "Conservative (prioritize safety)": 3,
            "Moderate (balanced approach)": 5,
            "Aggressive (seek growth)": 7,
            "Very aggressive (high risk, high reward)": 10
        }
    elif scale_type == "follow_through_rate":
        options = {
            "Rarely follow recommendations": 0.1,
            "Occasionally follow recommendations": 0.3,
            "Sometimes follow recommendations": 0.5,
            "Usually follow recommendations": 0.7,
            "Almost always follow recommendations": 0.9
        }
    elif scale_type == "override_frequency":
        options = {
            "Almost never override recommendations": 0.1,
            "Rarely override recommendations": 0.3,
            "Sometimes override recommendations": 0.5,
            "Often override recommendations": 0.7,
            "Almost always override recommendations": 0.9
        }
    elif scale_type == "app_usage_frequency":
        options = {
            "Rarely (few times a year)": 2,
            "Occasionally (monthly)": 4,
            "Regularly (weekly)": 6,
            "Frequently (several times a week)": 8,
            "Daily": 10
        }
    elif scale_type == "time_reviewing_recs":
        options = {
            "Very brief review (under 5 minutes)": 5,
            "Quick review (5-10 minutes)": 10,
            "Moderate review (10-20 minutes)": 20,
            "Thorough review (20-30 minutes)": 30,
            "Extensive review (over 30 minutes)": 45
        }
    elif scale_type == "rating_scale":
        options = {
            "Very poor": 1,
            "Poor": 2,
            "Average": 3,
            "Good": 4,
            "Excellent": 5
        }
    return options.get(option, 0)

st.title("AI-Assisted Financial Trust & Efficiency Analyzer (India)")
st.header("Tell us about yourself")

# User-friendly input methods using descriptive English options
trust_score_options = ["Highly skeptical", "Somewhat skeptical", "Neutral", "Somewhat trusting", "Highly trusting"]
trust_score_choice = st.selectbox("How much do you trust financial advice from AI?", options=trust_score_options)
trust_score = convert_to_numerical_value(trust_score_choice, "trust_score")

financial_literacy_options = ["Beginner (new to investing)", "Basic knowledge", "Intermediate", "Advanced", "Expert (finance professional)"]
financial_literacy_choice = st.selectbox("What is your level of financial knowledge?", options=financial_literacy_options)
financial_literacy = convert_to_numerical_value(financial_literacy_choice, "financial_literacy")

risk_tolerance_options = ["Very conservative (avoid losses at all costs)", "Conservative (prioritize safety)", 
                         "Moderate (balanced approach)", "Aggressive (seek growth)", "Very aggressive (high risk, high reward)"]
risk_tolerance_choice = st.selectbox("What is your investment risk tolerance?", options=risk_tolerance_options)
risk_tolerance = convert_to_numerical_value(risk_tolerance_choice, "risk_tolerance")

follow_through_options = ["Rarely follow recommendations", "Occasionally follow recommendations", 
                        "Sometimes follow recommendations", "Usually follow recommendations", "Almost always follow recommendations"]
follow_through_choice = st.selectbox("How often do you follow investment recommendations?", options=follow_through_options)
follow_through_rate = convert_to_numerical_value(follow_through_choice, "follow_through_rate")

override_options = ["Almost never override recommendations", "Rarely override recommendations", 
                  "Sometimes override recommendations", "Often override recommendations", "Almost always override recommendations"]
override_choice = st.selectbox("How often do you override or modify investment recommendations?", options=override_options)
override_frequency = convert_to_numerical_value(override_choice, "override_frequency")

app_usage_options = ["Rarely (few times a year)", "Occasionally (monthly)", "Regularly (weekly)", 
                    "Frequently (several times a week)", "Daily"]
app_usage_choice = st.selectbox("How often do you use financial apps?", options=app_usage_options)
app_usage_frequency = convert_to_numerical_value(app_usage_choice, "app_usage_frequency")

time_reviewing_options = ["Very brief review (under 5 minutes)", "Quick review (5-10 minutes)", 
                         "Moderate review (10-20 minutes)", "Thorough review (20-30 minutes)", "Extensive review (over 30 minutes)"]
time_reviewing_choice = st.selectbox("How much time do you typically spend reviewing investment recommendations?", options=time_reviewing_options)
time_reviewing_recs = convert_to_numerical_value(time_reviewing_choice, "time_reviewing_recs")

st.header("Rate your experience with financial advice tools")
rating_options = ["Very poor", "Poor", "Average", "Good", "Excellent"]

perceived_usefulness_choice = st.selectbox("How useful do you find investment recommendations?", options=rating_options)
perceived_usefulness = convert_to_numerical_value(perceived_usefulness_choice, "rating_scale")

perceived_accuracy_choice = st.selectbox("How accurate do you find investment predictions?", options=rating_options)
perceived_accuracy = convert_to_numerical_value(perceived_accuracy_choice, "rating_scale")

explanation_quality_choice = st.selectbox("How would you rate the quality of explanations for investment advice?", options=rating_options)
explanation_quality = convert_to_numerical_value(explanation_quality_choice, "rating_scale")

transparency_rating_choice = st.selectbox("How transparent do you find financial tools?", options=rating_options)
transparency_rating = convert_to_numerical_value(transparency_rating_choice, "rating_scale")

privacy_concerns_options = ["No concerns at all", "Minor concerns", "Moderate concerns", "Significant concerns", "Major concerns"]
privacy_concerns_choice = st.selectbox("How concerned are you about privacy when using financial tools?", options=privacy_concerns_options)
privacy_concerns = convert_to_numerical_value(privacy_concerns_choice, "rating_scale")

rec_type = st.selectbox("What kind of recommendations do you prefer?", options=["Low Risk", "Balanced", "High Risk"], index=1)
risk_level = st.selectbox("What is your current financial risk level?", options=["Low", "Medium", "High"], index=1)

if st.button("Get Personalized Recommendations"):
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
    
    st.subheader("Your Personalized Financial Analysis")
    st.markdown(f"**Trust Score:** {prediction_result['predicted_trust']}")
    st.markdown(f"**Predicted Financial Outcome:** {prediction_result['predicted_financial_outcome']}")
    segment_names = {
    1: "Confident Optimizers",  
    2: "Skeptical Cautious",    
    3: "Moderate Evaluators",   
    4: "Practical Skeptics"     
}
    st.markdown(f"**Your Investor Profile:** User Segment {prediction_result['user_segment']} ({segment_names[prediction_result['user_segment']]})")
    st.subheader("What This Means for You")
    st.markdown('''Confident Optimizers (Highest Trust, Best Outcome)
    You actively trust AI-assisted financial tools and leverage them for decision-making. Your financial outcomes tend to be better because of this high engagement.''')
    st.markdown('''Skeptical Pragmatists (Lowest Trust, Cautious Approach)
    You remain skeptical about AI-driven financial insights but still use them cautiously. You might validate recommendations through external sources before taking action.''')
    st.markdown('''Cautious Adopters (Moderate Trust, Careful Users)
    You see value in AI-assisted finance but prefer a careful approach. You may need more transparency or proof before fully trusting the system.''')
    st.markdown('''Neutral Observers (Balanced Trust, Passive Users)
    You neither fully trust nor distrust AI tools. You engage passively, possibly experimenting but not relying heavily on AI-driven insights.''')
    st.subheader("Recommended Strategy")
    st.markdown(f"**{prediction_result['recommended_strategy']}**")
    st.markdown(f"**Transparency Approach:** {prediction_result['transparency_approach']}")
    st.subheader("Investment Recommendations for Indian Market")
    for investment in prediction_result['investment_recommendations']:
        st.markdown(f"• {investment}")

st.markdown("---")
st.markdown("*This tool provides AI-driven financial analysis and investment recommendations specifically for the Indian market based on your personal preferences and financial behavior.*")