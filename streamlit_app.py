import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go  # Added for completeness, but not used in the "yellow only" version.
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt

def load_data():
    transactions_df = pd.read_parquet("transformed_transactions.parquet")
    users_df = pd.read_parquet("transformed_users.parquet")
    market_df = pd.read_parquet("transformed_market.parquet")
    network_df = pd.read_parquet("transformed_network.parquet")
    return transactions_df, users_df, market_df, network_df

transactions_df, users_df, market_df, network_df = load_data()

st.set_page_config(layout="wide")


st.title("Crypto Data Dashboard")

col1, col2 = st.columns(2)


with col1:
    # Transaction Volume Over Time
    st.subheader("Transaction Volume Over Time")
    transactions_df['date'] = transactions_df['timestamp'].dt.date
    daily_volume = transactions_df.groupby('date')['amount'].sum().reset_index()
    fig_volume = px.line(daily_volume, x='date', y='amount', color_discrete_sequence=["yellow"], title="Daily Transaction Volume")
    st.plotly_chart(fig_volume)

    # Transaction Type Distribution (Pie Chart)
    st.subheader("Transaction Type Distribution")
    transaction_type_counts = transactions_df['transaction_type'].value_counts()
    fig_type = px.pie(names=transaction_type_counts.index, values=transaction_type_counts.values, title="Transaction Type Distribution")
    st.plotly_chart(fig_type)

    # User Registration Trends (Bar Chart)
    st.subheader("User Registration Trends")
    users_df['registration_month'] = users_df['registration_date'].dt.to_period('M')
    monthly_registrations = users_df['registration_month'].value_counts().sort_index().reset_index()
    monthly_registrations.columns = ['registration_month', 'count']
    # Convert Period to string
    monthly_registrations['registration_month'] = monthly_registrations['registration_month'].astype(str)
    fig_user_registrations = px.line(monthly_registrations, x='registration_month', color_discrete_sequence=["yellow"], y='count',  title="Monthly User Registrations")
    st.plotly_chart(fig_user_registrations)


    # Top Traders by Volume
    st.subheader("Top Traders by Volume")
    top_traders = transactions_df.groupby('from_address')['amount_usd'].sum().nlargest(10).reset_index()
    fig_top_traders = px.bar(top_traders, x='from_address', y='amount_usd', color_discrete_sequence=["yellow"], title="Top 10 Traders by Volume (USD)")
    st.plotly_chart(fig_top_traders)

    # Average Transaction Fee by Token
    st.subheader("Average Transaction Fee by Token")
    avg_fee_by_token = transactions_df.groupby('token_symbol')['transaction_fee_usd'].mean().reset_index()
    fig_avg_fee = px.bar(avg_fee_by_token, x='token_symbol', y='transaction_fee_usd', color_discrete_sequence=["yellow"], title="Average Transaction Fee (USD) by Token")
    st.plotly_chart(fig_avg_fee)

    # User Activity Over Time (Active vs. Inactive)
    st.subheader("User Activity Over Time")
    user_activity = users_df.groupby(users_df['last_login_time'].dt.to_period('M'))['is_active'].value_counts().unstack().fillna(0).reset_index()
    user_activity.columns = ['month', 'Inactive', 'Active']
    fig_user_activity = px.line(user_activity, x=user_activity['month'].astype(str), y=['Active', 'Inactive'], title="User Activity Over Time")
    st.plotly_chart(fig_user_activity)

    # Correlation Heatmap (Transactions)
    st.subheader("Correlation Heatmap (Transactions)")
    numeric_trans_cols = transactions_df.select_dtypes(include=['number']).columns
    corr_trans = transactions_df[numeric_trans_cols].corr()
    fig_corr_trans = px.imshow(corr_trans, title="Correlation Heatmap (Transactions)")
    st.plotly_chart(fig_corr_trans)
            
    # Distribution of Account Balances (Users)
    st.subheader("Distribution of Account Balances (Users)")
    fig_account_balance = px.histogram(users_df, x='account_balance', title="Distribution of Account Balances")
    st.plotly_chart(fig_account_balance)

    # Trading Volume by Token
    st.subheader("Trading Volume by Token")
    volume_by_token = transactions_df.groupby('token_symbol')['amount_usd'].sum().reset_index()
    fig_volume_token = px.bar(volume_by_token,  color_discrete_sequence=["yellow"], x='token_symbol', y='amount_usd', title="Trading Volume (USD) by Token")
    st.plotly_chart(fig_volume_token)




with col2:

    # Top Countries by User Count
    st.subheader("Top Countries by User Count")
    country_counts = users_df['country'].value_counts().nlargest(10).reset_index()
    fig_country_users = px.bar(country_counts, color_discrete_sequence=["yellow"], x='country', y='count', title="Top 10 Countries by User Count")
    st.plotly_chart(fig_country_users)
    
    # Transaction Fees vs. Transaction Amount
    st.subheader("Transaction Fees vs. Transaction Amount")
    fig_fee_amount = px.scatter(transactions_df, x='amount_usd', y='transaction_fee_usd', title="Transaction Fees vs. Transaction Amount (USD)")
    st.plotly_chart(fig_fee_amount)

    # User Device Type Distribution
    st.subheader("User Device Type Distribution")
    device_counts = users_df['device_type'].value_counts().reset_index()
    fig_device_dist = px.pie(device_counts, values='count', names='device_type', title="User Device Type Distribution")
    st.plotly_chart(fig_device_dist)

    # Leverage Distribution in Transactions
    st.subheader("Leverage Distribution in Transactions")
    fig_leverage_dist = px.histogram(transactions_df, color_discrete_sequence=["yellow"], x='leverage', title="Leverage Distribution in Transactions")
    st.plotly_chart(fig_leverage_dist)

    # Rolling Average and Volatility (Transaction Volume)
    st.subheader("Rolling Average and Volatility (Transaction Volume) (Yellow Lines)")
    transactions_df['date'] = transactions_df['timestamp'].dt.date
    daily_volume_rolling = transactions_df.groupby('date')['amount'].sum().rolling(window=7).mean().reset_index()
    daily_volume_rolling['volatility'] = transactions_df.groupby('date')['amount'].sum().rolling(window=7).std().reset_index()['amount']
    fig_rolling_volume = px.line(daily_volume_rolling, x='date', y=['amount', 'volatility'], color_discrete_sequence=["yellow", "aqua"], title="Rolling Average and Volatility of Transaction Volume")
    st.plotly_chart(fig_rolling_volume)

    # User Segmentation Based on Activity
    st.subheader("User Segmentation Based on Activity (Yellow Pie)")
    users_df['days_since_last_login'] = (pd.to_datetime('now') - users_df['last_login_time']).dt.days
    def segment_user(days):
        if days < 7:
            return 'Active'
        elif days < 30:
            return 'Recent'
        else:
            return 'Inactive'
    users_df['activity_segment'] = users_df['days_since_last_login'].apply(segment_user)
    segment_counts = users_df['activity_segment'].value_counts().reset_index()
    fig_segmentation = px.pie(segment_counts, values='count', names='activity_segment', color_discrete_sequence=["yellow"], title="User Segmentation by Activity")
    st.plotly_chart(fig_segmentation)

    # Feature Importance (Transaction Data)
    st.subheader("Feature Importance (Transaction Data) (Yellow Bars)")
    numeric_trans_cols = transactions_df.select_dtypes(include=['number']).columns
    numeric_trans_cols = numeric_trans_cols.drop('amount')
    X = transactions_df[numeric_trans_cols].fillna(0)
    y = transactions_df['amount']
    model = RandomForestRegressor(random_state=42)
    model.fit(X, y)
    feature_importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    fig_feature_importance = px.bar(feature_importance.reset_index(), x='index', y=0, color_discrete_sequence=["yellow"], title="Feature Importance in Predicting Transaction Amount")
    st.plotly_chart(fig_feature_importance)

    # Time Series Analysis (Gas Prices)
    st.subheader("Time Series Analysis (Gas Prices)")
    gas_prices = network_df.set_index('network_timestamp')['average_gas_price']
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    plot_acf(gas_prices, ax=ax1, lags=30)
    plot_pacf(gas_prices, ax=ax2, lags=30)
    st.pyplot(fig)



