# 🏡 Real Estate Investment Advisor

A Machine Learning–powered Streamlit web application that helps users evaluate whether a property is a good investment and predicts its future price after 5 years based on real estate data.

🔗 **Live App:**

https://realestate-investment-advisor.streamlit.app/

# **Project Overview**

Investing in real estate requires analyzing multiple factors such as price, location, size, age of property, and amenities.
This project simplifies decision-making by using machine learning models to:

* ✅ Classify whether a property is a Good Investment
* 💰 Predict the Estimated Property Price after 5 Years
* 📊 Provide interactive Market Insights using visual analytics

# Key Features
⚡ **Quick Investment Predictor**

* User-friendly form to enter property details
* Instant investment recommendation
* Confidence score for prediction
* Future price estimation

# Market Insights Dashboard

* City-wise average price (Bar chart)
* Price distribution by BHK (Box plot)
* Property type contribution (Donut chart)
* Size vs price correlation (Scatter plot)
* Age vs price trend (Line chart)
* City-wise price distribution (Violin plot)

# UI & UX

* Fully **dark-themed professional UI**
* Modern gradient action button
* No sidebar distractions
* Responsive and clean layout

# Machine Learning Models
 🔹 **Classification Model**

* **Algorithm:** Logistic Regression
* **Target:** Good_Investment
* **Metrics Used:**
  * Accuracy
  * Precision
  * Recall
  * ROC-AUC

🔹 **Regression Model**

* **Algorithm:** XGBoost Regressor
* **Target:** Future_Price_5Y
* **Metrics Used:**
  * RMSE
  * MAE
  * R² Score
 
# Tech Stack

* **Frontend:** Streamlit
* **Backend:** Python
* **ML Libraries:** scikit-learn, XGBoost
* **Data Processing:** Pandas, NumPy
* **Visualization:** Plotly, Matplotlib
* **Model Persistence:** Joblib
* **Deployment:** Streamlit Cloud
* **Version Control:** GitHub

# 📂 Project Structure
```
RealEstateAdvisor/
│
├── app.py
├── requirements.txt
├── classification_model.pkl
├── regression_model.pkl
├── feature_columns.pkl
├── README.md
```

# Installation & Local Run
1️⃣ **Clone the repository**
```
git clone https://github.com/your-username/real-estate-investment-advisor.git
cd real-estate-investment-advisor
```
2️⃣ **Install dependencies**
```
pip install -r requirements.txt
```
3️⃣ Run the app
```
streamlit run app.py
```
# Deployment

The application is deployed using Streamlit Cloud and automatically rebuilds on every GitHub update using requirements.txt.

# Performance & Optimization

* Cached datasets and models using Streamlit caching
* Lightweight inference (no retraining during runtime)
* Optimized UI rendering with tab-based layout
* Stable and smooth performance on cloud deployment

# Academic & Learning Outcomes

* End-to-end ML pipeline design
* Feature engineering & model evaluation
* ML model deployment
* Real-world UI/UX considerations
* Handling cloud dependency issues
* Production-level Streamlit development

# Author

**Vadla Shiva Kumar**
**GitHub:** https://github.com/shivavadla931
