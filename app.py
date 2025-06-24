import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import requests
import time

# Page config
st.set_page_config(
    page_title="MSTR Options Tracker",
    page_icon="📈",
    layout="wide"
)

class PolygonDataFetcher:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.polygon.io"
    
    def get_stock_price(self, symbol="MSTR"):
        """Get current stock price"""
        try:
            url = f"{self.base_url}/v2/aggs/ticker/{symbol}/prev"
            params = {"apikey": self.api_key}
            response = requests.get(url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                if data.get("results"):
                    return data["results"][0]["c"]  # closing price
            return None
        except Exception as e:
            st.error(f"Error fetching stock price: {e}")
            return None
    
    def get_historical_data(self, symbol="MSTR", days=30):
        """Get historical stock data"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            url = f"{self.base_url}/v2/aggs/ticker/{symbol}/range/1/day/{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"
            params = {"apikey": self.api_key}
            response = requests.get(url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                if data.get("results"):
                    df = pd.DataFrame(data["results"])
                    df['date'] = pd.to_datetime(df['t'], unit='ms')
                    df['price'] = df['c']  # closing price
                    return df[['date', 'price']]
            return None
        except Exception as e:
            st.error(f"Error fetching historical data: {e}")
            return None

def calculate_black_scholes_put(S, K, T, r, sigma):
    """Simplified Black-Scholes put option pricing"""
    from scipy.stats import norm
    import math
    
    d1 = (math.log(S/K) + (r + sigma**2/2)*T) / (sigma*math.sqrt(T))
    d2 = d1 - sigma*math.sqrt(T)
    
    put_price = K*math.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)
    put_delta = -norm.cdf(-d1)
    
    return put_price, put_delta

def simulate_options_data(stock_data, target_deltas=[0.05, 0.10]):
    """Simulate options prices based on stock movement"""
    if stock_data is None or stock_data.empty:
        return None
    
    options_data = stock_data.copy()
    
    # Options parameters
    r = 0.05  # risk-free rate
    sigma = 0.6  # implied volatility for MSTR
    T = 30/365  # 30 days to expiration
    
    for delta_target in target_deltas:
        option_prices = []
        
        for _, row in stock_data.iterrows():
            S = row['price']  # Current stock price
            
            # Find strike for target delta using approximation
            # For puts, delta is negative, so we target -delta_target
            K = S * np.exp(0.5 * sigma**2 * T + sigma * np.sqrt(T) * 
                          (np.log(delta_target) if delta_target > 0.5 else -np.log(1-delta_target)))
            
            # Calculate option price
            try:
                put_price, _ = calculate_black_scholes_put(S, K, T, r, sigma)
                # Add some realistic noise
                put_price *= (1 + np.random.normal(0, 0.1))
                option_prices.append(max(put_price, 0.01))  # Minimum price
            except:
                option_prices.append(1.0)  # Fallback price
        
        options_data[f'{int(delta_target*100)}_delta_put'] = option_prices
    
    return options_data

def main():
    st.title("📈 MSTR Options Tracker")
    st.markdown("*Cloud-based options tracking with real market data*")
    
    # Sidebar for API key
    st.sidebar.header("🔑 Configuration")
    
    api_key = st.sidebar.text_input(
        "Polygon.io API Key", 
        type="password",
        help="Get your free API key at polygon.io"
    )
    
    if not api_key:
        st.warning("👆 Please enter your Polygon.io API key in the sidebar to get started!")
        st.markdown("""
        ### 🚀 Quick Start:
        1. Get your free API key at [polygon.io](https://polygon.io)
        2. Paste it in the sidebar
        3. Click "Fetch Data" below
        
        ### 📊 What This App Does:
        - Fetches real MSTR stock prices
        - Simulates 5-delta and 10-delta put option prices
        - Creates interactive charts
        - Exports data to CSV
        """)
        return
    
    # Initialize data fetcher
    fetcher = PolygonDataFetcher(api_key)
    
    # Main interface
    col1, col2, col3 = st.columns(3)
    
    # Current price
    with col1:
        if st.button("🔄 Get Current MSTR Price", type="primary"):
            with st.spinner("Fetching current price..."):
                price = fetcher.get_stock_price()
                if price:
                    st.metric("MSTR Current Price", f"${price:.2f}")
                else:
                    st.error("Could not fetch current price")
    
    # Historical data section
    st.header("📈 Historical Analysis")
    
    col1, col2 = st.columns(2)
    with col1:
        days = st.selectbox("Historical Period", [7, 14, 30, 60], index=2)
    
    with col2:
        if st.button("📊 Fetch Historical Data", type="primary"):
            with st.spinner(f"Fetching {days} days of data..."):
                # Get stock data
                stock_data = fetcher.get_historical_data(days=days)
                
                if stock_data is not None and not stock_data.empty:
                    # Simulate options data
                    options_data = simulate_options_data(stock_data)
                    
                    if options_data is not None:
                        st.session_state['data'] = options_data
                        st.success(f"✅ Loaded {len(options_data)} days of data!")
                    else:
                        st.error("Could not simulate options data")
                else:
                    st.error("Could not fetch historical data. Check your API key.")
    
    # Display charts if data exists
    if 'data' in st.session_state:
        data = st.session_state['data']
        
        # Create interactive chart
        fig = go.Figure()
        
        # Add stock price
        fig.add_trace(go.Scatter(
            x=data['date'],
            y=data['price'],
            name='MSTR Stock Price',
            line=dict(color='blue', width=3),
            yaxis='y2'
        ))
        
        # Add options prices
        colors = ['red', 'orange']
        deltas = [10, 5]
        
        for i, delta in enumerate(deltas):
            col_name = f'{delta}_delta_put'
            if col_name in data.columns:
                fig.add_trace(go.Scatter(
                    x=data['date'],
                    y=data[col_name],
                    name=f'{delta}-Delta Put',
                    line=dict(color=colors[i], width=2),
                    yaxis='y'
                ))
        
        # Update layout
        fig.update_layout(
            title="MSTR Options vs Stock Price",
            xaxis_title="Date",
            yaxis=dict(
                title="Option Price ($)",
                side="left"
            ),
            yaxis2=dict(
                title="Stock Price ($)",
                side="right",
                overlaying="y"
            ),
            height=600,
            hovermode='x unified',
            legend=dict(x=0, y=1)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Data summary
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Current Stock Price", 
                f"${data['price'].iloc[-1]:.2f}",
                f"{data['price'].iloc[-1] - data['price'].iloc[0]:.2f}"
            )
        
        if '10_delta_put' in data.columns:
            with col2:
                st.metric(
                    "10-Delta Put", 
                    f"${data['10_delta_put'].iloc[-1]:.2f}",
                    f"{data['10_delta_put'].iloc[-1] - data['10_delta_put'].iloc[0]:.2f}"
                )
        
        if '5_delta_put' in data.columns:
            with col3:
                st.metric(
                    "5-Delta Put", 
                    f"${data['5_delta_put'].iloc[-1]:.2f}",
                    f"{data['5_delta_put'].iloc[-1] - data['5_delta_put'].iloc[0]:.2f}"
                )
        
        # Data table
        st.subheader("📋 Raw Data")
        st.dataframe(data, use_container_width=True)
        
        # Download button
        csv = data.to_csv(index=False)
        st.download_button(
            label="💾 Download Data as CSV",
            data=csv,
            file_name=f"mstr_options_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    
    # Footer
    st.markdown("---")
    st.markdown("*Built with Streamlit • Data from Polygon.io • Options prices simulated using Black-Scholes*")

if __name__ == "__main__":
    main()