import streamlit as st
import pandas as pd
import json
import plotly.express as px

# Set page config
st.set_page_config(page_title="ITDR Data Explorer", layout="wide")

st.title("ITDR Data & Alert Explorer")

@st.cache_data
def load_data():
    data = []
    try:
        with open("../sample_logs.jsonl", 'r') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
    except FileNotFoundError:
        return pd.DataFrame()
    
    # Flatten for display
    processed_data = []
    for entry in data:
        flat_entry = {
            "timestamp": dict(entry).get("timestamp"),
            "event_type": dict(entry).get("eventType"),
            "user": dict(entry).get("identity", {}).get("upn"),
            "ip": dict(entry).get("location", {}).get("ip"),
            "country": dict(entry).get("location", {}).get("country"),
            "city": dict(entry).get("location", {}).get("city"),
            "lat": None, # In a real app we'd geoip this
            "lon": None,
            "status": dict(entry).get("status"),
            "is_attack": dict(entry).get("label", {}).get("isAttack", False),
            "attack_type": dict(entry).get("label", {}).get("attackType"),
            "raw": json.dumps(entry)
        }
        processed_data.append(flat_entry)
        
    df = pd.DataFrame(processed_data)
    if not df.empty:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

df = load_data()

if df.empty:
    st.error("No data found. Please run `generate_data.py` first.")
else:
    # Sidebar filters
    st.sidebar.header("Filters")
    show_attacks_only = st.sidebar.checkbox("Show Attacks Only", value=False)
    
    if show_attacks_only:
        df_display = df[df['is_attack'] == True]
    else:
        df_display = df

    # distinct_users = len(df['user'].unique())
    # distinct_ips = len(df['ip'].unique())
    # attack_count = len(df[df['is_attack']==True])
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Events", len(df))
    col2.metric("Attacks", len(df[df['is_attack']==True]))
    col3.metric("Normal", len(df[df['is_attack']==False]))

    st.subheader("Event Timeline")
    # Simple line chart of events over time
    if not df_display.empty:
        df_chart = df_display.set_index('timestamp').resample('5min').size().rename('count')
        st.line_chart(df_chart)

    st.subheader("Log Data")
    st.dataframe(
        df_display[['timestamp', 'event_type', 'user', 'ip', 'country', 'status', 'is_attack', 'attack_type']],
        width="stretch"
    )
    
    st.subheader("Attack Type Distribution")
    if not df[df['is_attack']==True].empty:
        fig = px.pie(df[df['is_attack']==True], names='attack_type')
        st.plotly_chart(fig)
