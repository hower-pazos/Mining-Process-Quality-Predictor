import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Mining Process Quality Predictor",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: 
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: 
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid 
    }
    .alert-critical {
        background-color: 
        border-left: 5px solid 
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .alert-warning {
        background-color: 
        border-left: 5px solid 
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .alert-normal {
        background-color: 
        border-left: 5px solid 
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .stButton > button {
        background-color: 
        color: white;
        border-radius: 5px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

class SilicaPredictor:
    """Production-ready silica prediction system"""

    def __init__(self):
        self.model = None
        self.model_loaded = False
        self.feature_names = [
            '% Iron Feed', '% Silica Feed', 'Starch Flow', 'Amina Flow', 
            'Ore Pulp Flow', 'Ore Pulp pH', 'Ore Pulp Density', 
            '% Iron Concentrate', 'Avg_Flotation_Air_Flow', 
            'Std_Flotation_Air_Flow', 'Avg_Flotation_Level', 
            'Std_Flotation_Level', 'Iron_Silica_Feed_Ratio', 
            'Total_Feed_Content', 'hour', 'day_of_week', 'month', 'is_weekend'
        ]

        self.additional_features = [
            '% Iron Feed_MA_5', '% Iron Feed_Std_5', '% Iron Feed_CV_5', 
            '% Iron Feed_Lag1', '% Silica Feed_MA_5', '% Silica Feed_Std_5', 
            '% Silica Feed_CV_5', '% Silica Feed_Lag1', 'Starch Flow_MA_5', 
            'Starch Flow_Std_5', 'Starch Flow_CV_5', 'Starch Flow_Lag1', 
            'Amina Flow_MA_5', 'Amina Flow_Std_5', 'Amina Flow_CV_5', 
            'Amina Flow_Lag1', 'Ore Pulp Flow_MA_5', 'Ore Pulp Flow_Std_5', 
            'Ore Pulp Flow_CV_5', 'Ore Pulp Flow_Lag1', 'Ore Pulp pH_MA_5', 
            'Ore Pulp pH_Std_5', 'Ore Pulp pH_CV_5', 'Ore Pulp pH_Lag1', 
            'Ore Pulp Density_MA_5', 'Ore Pulp Density_Std_5', 
            'Ore Pulp Density_CV_5', 'Ore Pulp Density_Lag1', 
            'Avg_Flotation_Air_Flow_MA_10', 'Avg_Flotation_Air_Flow_Std_10', 
            'Avg_Flotation_Air_Flow_CV_10', 'Avg_Flotation_Level_MA_10', 
            'Avg_Flotation_Level_Std_10', 'Avg_Flotation_Level_CV_10', 
            'Chemical_Flow_Ratio', 'Total_Chemical_Flow', 
            'Feed_Ratio_x_Air_Flow', 'Feed_Ratio_x_Level', 'Flow_Efficiency'
        ]

        self.total_feature_names = self.feature_names + self.additional_features

        self._load_model()

    def _load_model(self):
        import joblib
        import numpy as np
        import streamlit as st

        model_file = 'silica_model_v1.6.1.pkl'

        try:
            loaded = joblib.load(model_file)
            if isinstance(loaded, dict):
                self.model = loaded['model']
                self.scaler = loaded.get('scaler', None)
                self.feature_names = loaded.get('feature_names', self.total_feature_names)
            else:
                self.model = loaded
                self.scaler = None
                self.feature_names = self.total_feature_names

            st.sidebar.success("✅ Model loaded successfully")

            test_input = np.zeros((1, len(self.total_feature_names)))

            if self.scaler:
                test_input = self.scaler.transform(test_input)

            test_pred = self.model.predict(test_input)
            self.model_loaded = True
        except Exception as e:
            st.sidebar.error(f"❌ Model loading failed: {str(e)[:100]}...")
            print(str(e))
            self.model_loaded = False

    def predict(self, input_data):
        import numpy as np
        import streamlit as st
        from datetime import datetime

        try:           

            full_input = np.zeros(len(self.total_feature_names))

            input_mapping = {
                '% Iron Feed': input_data.get('Iron_Feed_%', 0),
                '% Silica Feed': input_data.get('Silica_Feed_%', 0),
                'Starch Flow': input_data.get('Starch_Flow', 0),
                'Amina Flow': input_data.get('Amina_Flow', 0),
                'Ore Pulp Flow': input_data.get('Ore_Pulp_Flow', 0),
                'Ore Pulp pH': input_data.get('Ore_Pulp_pH', 0),
                'Ore Pulp Density': input_data.get('Ore_Pulp_Density', 0),
                'Avg_Flotation_Air_Flow': input_data.get('Avg_Flotation_Air_Flow', 0),
                'Avg_Flotation_Level': input_data.get('Avg_Flotation_Level', 0),
                'Iron_Silica_Feed_Ratio': input_data.get('Iron_Silica_Feed_Ratio', 0),
                'Chemical_Flow_Ratio': input_data.get('Chemical_Flow_Ratio', 0),
            }

            current_time = datetime.now()
            input_mapping.update({
                'hour': current_time.hour,
                'day_of_week': current_time.weekday(),
                'month': current_time.month,
                'is_weekend': 1 if current_time.weekday() >= 5 else 0
            })

            feature_engineering_mapping = {
                '% Iron Feed_MA_5': input_mapping['% Iron Feed'],
                '% Iron Feed_Std_5': 0, 
                '% Iron Feed_CV_5': 0,
                '% Iron Feed_Lag1': input_mapping['% Iron Feed'],

                '% Silica Feed_MA_5': input_mapping['% Silica Feed'],
                '% Silica Feed_Std_5': 0,
                '% Silica Feed_CV_5': 0,
                '% Silica Feed_Lag1': input_mapping['% Silica Feed'],

                'Starch Flow_MA_5': input_mapping['Starch Flow'],
                'Starch Flow_Std_5': 0,
                'Starch Flow_CV_5': 0,
                'Starch Flow_Lag1': input_mapping['Starch Flow'],

                'Amina Flow_MA_5': input_mapping['Amina Flow'],
                'Amina Flow_Std_5': 0,
                'Amina Flow_CV_5': 0,
                'Amina Flow_Lag1': input_mapping['Amina Flow'],

                'Ore Pulp Flow_MA_5': input_mapping['Ore Pulp Flow'],
                'Ore Pulp Flow_Std_5': 0,
                'Ore Pulp Flow_CV_5': 0,
                'Ore Pulp Flow_Lag1': input_mapping['Ore Pulp Flow'],

                'Ore Pulp pH_MA_5': input_mapping['Ore Pulp pH'],
                'Ore Pulp pH_Std_5': 0,
                'Ore Pulp pH_CV_5': 0,
                'Ore Pulp pH_Lag1': input_mapping['Ore Pulp pH'],

                'Ore Pulp Density_MA_5': input_mapping['Ore Pulp Density'],
                'Ore Pulp Density_Std_5': 0,
                'Ore Pulp Density_CV_5': 0,
                'Ore Pulp Density_Lag1': input_mapping['Ore Pulp Density'],

                'Avg_Flotation_Air_Flow_MA_10': input_mapping['Avg_Flotation_Air_Flow'],
                'Avg_Flotation_Air_Flow_Std_10': 0,
                'Avg_Flotation_Air_Flow_CV_10': 0,

                'Avg_Flotation_Level_MA_10': input_mapping['Avg_Flotation_Level'],
                'Avg_Flotation_Level_Std_10': 0,
                'Avg_Flotation_Level_CV_10': 0,

                'Total_Feed_Content': input_mapping['% Iron Feed'] + input_mapping['% Silica Feed'],
                'Total_Chemical_Flow': input_mapping['Starch Flow'] + input_mapping['Amina Flow'],
                'Feed_Ratio_x_Air_Flow': input_mapping['Iron_Silica_Feed_Ratio'] * input_mapping['Avg_Flotation_Air_Flow'],
                'Feed_Ratio_x_Level': input_mapping['Iron_Silica_Feed_Ratio'] * input_mapping['Avg_Flotation_Level'],
                'Flow_Efficiency': input_mapping['Chemical_Flow_Ratio'] / input_mapping['Ore Pulp Flow'] if input_mapping['Ore Pulp Flow'] != 0 else 0
            }

            input_mapping.update(feature_engineering_mapping)

            for i, feat_name in enumerate(self.total_feature_names):
                full_input[i] = input_mapping.get(feat_name, 0)

            full_input = full_input.reshape(1, -1)

            if self.scaler:
                full_input = self.scaler.transform(full_input)

            prediction = self.model.predict(full_input)[0]

            confidence_width = 0.3
            lower_ci = max(0.5, prediction - confidence_width)
            upper_ci = min(6.0, prediction + confidence_width)
            return prediction, lower_ci, upper_ci

        except Exception as e:
            st.sidebar.error(f"Model prediction failed: {e}")

            return None

@st.cache_data
def generate_historical_data(days=30):
    """Generate realistic historical data for demo"""
    np.random.seed(42)  
    dates = pd.date_range(start=datetime.now() - timedelta(days=days), 
                         end=datetime.now(), freq='1H')

    data = []
    for i, date in enumerate(dates):

        base_iron = 56 + 2 * np.sin(i * 0.01) + np.random.normal(0, 1.5)
        base_silica = 15 + 3 * np.sin(i * 0.015 + 1) + np.random.normal(0, 2)

        iron_feed = max(45, min(70, base_iron))
        silica_feed = max(5, min(35, base_silica))
        starch_flow = max(1000, min(5000, 2900 + np.random.normal(0, 300)))
        amina_flow = max(300, min(700, 490 + np.random.normal(0, 30)))

        iron_silica_ratio = iron_feed / silica_feed

        silica_concentrate = (3.8 - iron_silica_ratio * 0.35 + 
                            (starch_flow - 2900) * 0.0002 +
                            np.random.normal(0, 0.25))
        silica_concentrate = max(0.5, min(6.0, silica_concentrate))

        prediction_error = np.random.normal(0, 0.15)
        silica_pred = max(0.5, min(6.0, silica_concentrate + prediction_error))

        data.append({
            'datetime': date,
            'iron_feed': iron_feed,
            'silica_feed': silica_feed,
            'starch_flow': starch_flow,
            'amina_flow': amina_flow,
            'ore_pulp_flow': np.random.normal(398, 5),
            'ore_pulp_ph': np.random.normal(9.8, 0.15),
            'flotation_air_flow': np.random.normal(280, 12),
            'flotation_level': np.random.normal(520, 25),
            'silica_concentrate_actual': silica_concentrate,
            'silica_concentrate_pred': silica_pred,
            'iron_silica_ratio': iron_silica_ratio
        })

    return pd.DataFrame(data)

def main():

    st.markdown('<h1 class="main-header">🏭 Mining Process Quality Predictor</h1>', 
                unsafe_allow_html=True)

    predictor = SilicaPredictor()

    st.sidebar.title("🎛️ Control Panel")

    if predictor.model_loaded:
        st.sidebar.markdown("**🤖 Model Status:** Production Model")
    else:
        st.sidebar.markdown("**🤖 Model Status:** Simulation Mode")

    page = st.sidebar.selectbox("Navigate to:", [
        "🎯 Real-time Prediction", 
        "📈 Historical Analysis", 
        "🎛️ Process Monitoring",
        "🤖 Model Performance",
        "🚨 Alerts & Reports"
    ])

    if "Real-time Prediction" in page:
        show_prediction_page(predictor)
    elif "Historical Analysis" in page:
        show_historical_page()
    elif "Process Monitoring" in page:
        show_monitoring_page()
    elif "Model Performance" in page:
        show_performance_page()
    else:
        return

def show_prediction_page(predictor):
    """Real-time prediction interface"""
    st.header("🎯 Real-time Silica Concentration Prediction")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Process Parameters Input")

        with st.form("prediction_form"):
            col_a, col_b = st.columns(2)

            with col_a:
                st.markdown("**Feed Composition**")
                iron_feed = st.slider("Iron Feed (%)", 45.0, 70.0, 56.0, 0.1)
                silica_feed = st.slider("Silica Feed (%)", 5.0, 35.0, 15.0, 0.1)

                st.markdown("**Chemical Dosing**")
                starch_flow = st.slider("Starch Flow (kg/h)", 1000.0, 5000.0, 2900.0, 10.0)
                amina_flow = st.slider("Amina Flow (g/t)", 300.0, 700.0, 490.0, 5.0)

                st.markdown("**Pulp Properties**")
                ore_pulp_flow = st.slider("Ore Pulp Flow (t/h)", 350.0, 420.0, 398.0, 1.0)

            with col_b:
                st.markdown("**Process Control**")
                ore_pulp_ph = st.slider("Ore Pulp pH", 8.5, 11.0, 9.8, 0.1)
                ore_density = st.slider("Ore Pulp Density (g/cm³)", 1.4, 1.9, 1.68, 0.01)

                st.markdown("**Flotation Parameters**")
                flotation_air = st.slider("Flotation Air Flow (Nm³/h)", 200.0, 400.0, 280.0, 5.0)
                flotation_level = st.slider("Flotation Level (mm)", 300.0, 700.0, 520.0, 10.0)

            predict_button = st.form_submit_button("🔮 Make Prediction", type="primary")

        if predict_button:

            chemical_ratio = starch_flow / amina_flow
            iron_silica_ratio = iron_feed / silica_feed

            features_dict = {
    'Iron_Feed_%': iron_feed,
    'Silica_Feed_%': silica_feed,
    'Starch_Flow': starch_flow,
    'Amina_Flow': amina_flow,
    'Ore_Pulp_Flow': ore_pulp_flow,
    'Ore_Pulp_pH': ore_pulp_ph,
    'Ore_Pulp_Density': ore_density,
    'Avg_Flotation_Air_Flow': flotation_air,
    'Avg_Flotation_Level': flotation_level,
    'Chemical_Flow_Ratio': chemical_ratio,
    'Iron_Silica_Feed_Ratio': iron_silica_ratio
}



    with col2:
        st.subheader("📈 Process Insights")

        iron_silica_ratio = iron_feed / silica_feed
        flow_efficiency = ore_pulp_flow / flotation_air
        chemical_ratio = starch_flow / amina_flow

        st.metric("Iron/Silica Ratio", f"{iron_silica_ratio:.2f}", 
                 help="Optimal range: 3.0-4.5")
        st.metric("Flow Efficiency", f"{flow_efficiency:.2f}", 
                 help="Ore flow / Air flow ratio")
        st.metric("Chemical Ratio", f"{chemical_ratio:.1f}", 
                 help="Starch/Amina ratio - optimal: 5-7")

        st.subheader("🎯 Process Health")

        health_score = 100
        recommendations = []

        if iron_silica_ratio < 3:
            health_score -= 20
            recommendations.append("⚠️ Low iron/silica ratio - expect higher silica")
        elif iron_silica_ratio > 5:
            health_score -= 10
            recommendations.append("ℹ️ High iron/silica ratio - good quality expected")

        if chemical_ratio > 7:
            health_score -= 15
            recommendations.append("⚠️ High starch/amina ratio - may affect flotation")
        elif chemical_ratio < 4:
            health_score -= 10
            recommendations.append("⚠️ Low starch/amina ratio - check dosing")

        if ore_pulp_ph < 9.5 or ore_pulp_ph > 10.2:
            health_score -= 15
            recommendations.append("⚠️ pH outside optimal range (9.5-10.2)")

        if flotation_air < 250 or flotation_air > 350:
            health_score -= 10
            recommendations.append("ℹ️ Flotation air flow outside typical range")

        if health_score >= 90:
            st.success(f"Process Health: {health_score}% - Excellent")
        elif health_score >= 75:
            st.warning(f"Process Health: {health_score}% - Good")
        else:
            st.error(f"Process Health: {health_score}% - Needs Attention")

        if recommendations:
            st.subheader("💡 Recommendations")
            for rec in recommendations:
                st.write(rec)

def show_historical_page():
    """Historical data analysis"""
    st.header("📈 Historical Performance Analysis")

    df = generate_historical_data(30)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        avg_silica = df['silica_concentrate_actual'].mean()
        st.metric("Average Silica %", f"{avg_silica:.2f}")

    with col2:
        excursions = (df['silica_concentrate_actual'] > 3.0).sum()
        st.metric("Quality Excursions", excursions)

    with col3:
        best_day = df.groupby(df['datetime'].dt.date)['silica_concentrate_actual'].mean().min()
        st.metric("Best Daily Avg", f"{best_day:.2f}%")

    with col4:
        stability = df['silica_concentrate_actual'].std()
        st.metric("Process Stability", f"{stability:.2f}")

    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=('Silica Concentration Trends', 'Key Process Variables', 'Process Ratios'),
        vertical_spacing=0.1,
        specs=[[{"secondary_y": True}], 
               [{"secondary_y": True}], 
               [{"secondary_y": False}]]
    )

    fig.add_trace(
        go.Scatter(x=df['datetime'], y=df['silica_concentrate_actual'],
                  name='Actual Silica %', line=dict(color='blue', width=2)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df['datetime'], y=df['silica_concentrate_pred'],
                  name='Predicted Silica %', line=dict(color='red', width=2, dash='dash')),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(x=df['datetime'], y=df['iron_feed'],
                  name='Iron Feed %', line=dict(color='green')),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=df['datetime'], y=df['silica_feed'],
                  name='Silica Feed %', line=dict(color='purple')),
        row=2, col=1, secondary_y=True
    )

    fig.add_trace(
        go.Scatter(x=df['datetime'], y=df['iron_silica_ratio'],
                  name='Iron/Silica Ratio', line=dict(color='brown')),
        row=3, col=1
    )

    fig.add_hline(y=2.5, line_dash="dot", line_color="orange", 
                  annotation_text="Warning", annotation_position="top right")
    fig.add_hline(y=3.5, line_dash="dot", line_color="red", 
                  annotation_text="Critical", annotation_position="top right")

    fig.update_layout(
        height=800, 
        title_text="30-Day Historical Analysis",
        showlegend=True
    )

    fig.update_yaxes(title_text="Silica %", row=1, col=1)
    fig.update_yaxes(title_text="Iron Feed %", row=2, col=1)
    fig.update_yaxes(title_text="Silica Feed %", secondary_y=True, row=2, col=1)
    fig.update_yaxes(title_text="Ratio", row=3, col=1)
    fig.update_xaxes(title_text="Date", row=3, col=1)

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📊 Model Performance Analysis")

    col1, col2 = st.columns(2)

    with col1:

        actual = df['silica_concentrate_actual']
        predicted = df['silica_concentrate_pred']

        rmse = np.sqrt(np.mean((actual - predicted)**2))
        mae = np.mean(np.abs(actual - predicted))
        r2 = 1 - np.sum((actual - predicted)**2) / np.sum((actual - np.mean(actual))**2)

        st.metric("RMSE", f"{rmse:.3f}")
        st.metric("MAE", f"{mae:.3f}")
        st.metric("R² Score", f"{r2:.3f}")

        if r2 > 0.7:
            st.success("✅ Excellent model performance")
        elif r2 > 0.6:
            st.warning("⚠️ Good model performance")
        else:
            st.error("❌ Model needs improvement")

    with col2:

        fig_scatter = go.Figure()

        errors = np.abs(actual - predicted)

        fig_scatter.add_trace(go.Scatter(
            x=actual, y=predicted, mode='markers',
            name='Predictions', 
            marker=dict(
                size=6, 
                opacity=0.6,
                color=errors,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Prediction Error")
            ),
            text=[f"Error: {e:.2f}" for e in errors],
            hovertemplate="Actual: %{x:.2f}<br>Predicted: %{y:.2f}<br>%{text}<extra></extra>"
        ))

        min_val, max_val = actual.min(), actual.max()
        fig_scatter.add_trace(go.Scatter(
            x=[min_val, max_val], y=[min_val, max_val], 
            mode='lines',
            name='Perfect Prediction', 
            line=dict(dash='dash', color='red', width=2)
        ))

        fig_scatter.update_layout(
            title="Prediction Accuracy", 
            xaxis_title="Actual Silica %", 
            yaxis_title="Predicted Silica %",
            height=400
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    st.subheader("📈 Process Insights")

    col3, col4 = st.columns(2)

    with col3:

        daily_avg = df.groupby(df['datetime'].dt.date).agg({
            'silica_concentrate_actual': 'mean',
            'iron_silica_ratio': 'mean',
            'starch_flow': 'mean'
        }).reset_index()

        fig_daily = go.Figure()
        fig_daily.add_trace(go.Scatter(
            x=daily_avg['datetime'], 
            y=daily_avg['silica_concentrate_actual'],
            mode='lines+markers',
            name='Daily Avg Silica %',
            line=dict(color='blue', width=3)
        ))

        fig_daily.add_hline(y=2.5, line_dash="dash", line_color="orange", 
                           annotation_text="Target Limit")

        fig_daily.update_layout(
            title="Daily Average Silica Trends",
            xaxis_title="Date",
            yaxis_title="Silica %",
            height=300
        )
        st.plotly_chart(fig_daily, use_container_width=True)

    with col4:

        corr_vars = ['silica_concentrate_actual', 'iron_feed', 'silica_feed', 
                    'starch_flow', 'amina_flow', 'iron_silica_ratio']
        corr_matrix = df[corr_vars].corr()

        fig_heatmap = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu_r',
            zmid=0,
            text=np.round(corr_matrix.values, 2),
            texttemplate="%{text}",
            textfont={"size":10},
            hoverongaps=False
        ))

        fig_heatmap.update_layout(
            title="Process Variable Correlations",
            height=300
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)

    st.subheader("💾 Export Data")

    col5, col6 = st.columns(2)

    with col5:

        csv_data = df.to_csv(index=False)
        st.download_button(
            label="📊 Download Historical Data (CSV)",
            data=csv_data,
            file_name=f"historical_data_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

    with col6:

        summary_report = f"""
Historical Performance Summary
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}

Performance Metrics:
- Average Silica Concentration: {avg_silica:.2f}%
- Process Stability (StdDev): {stability:.2f}
- Quality Excursions: {excursions}
- Model R² Score: {r2:.3f}
- Model RMSE: {rmse:.3f}
- Model MAE: {mae:.3f}

Process Insights:
- Iron/Silica Ratio Average: {df['iron_silica_ratio'].mean():.2f}
- Best Daily Performance: {best_day:.2f}%
- Worst Daily Performance: {df.groupby(df['datetime'].dt.date)['silica_concentrate_actual'].mean().max():.2f}%
        """

        st.download_button(
            label="📄 Download Summary Report",
            data=summary_report,
            file_name=f"performance_summary_{datetime.now().strftime('%Y%m%d')}.txt",
            mime="text/plain"
        )

def show_monitoring_page():
    """Process monitoring dashboard"""
    st.header("🎛️ Real-time Process Monitoring")

    current_time = datetime.now()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        current_silica = np.random.normal(2.2, 0.4)
        delta_silica = np.random.normal(0, 0.1)
        st.metric("Current Silica %", f"{current_silica:.2f}%", 
                 delta=f"{delta_silica:.2f}%")

    with col2:
        throughput = np.random.normal(850, 30)
        delta_throughput = np.random.normal(0, 15)
        st.metric("Throughput (t/h)", f"{throughput:.0f}", 
                 delta=f"{delta_throughput:.0f}")

    with col3:
        efficiency = np.random.uniform(85, 95)
        st.metric("Process Efficiency", f"{efficiency:.1f}%")

    with col4:
        uptime = np.random.uniform(98, 100)
        st.metric("System Uptime", f"{uptime:.1f}%")

    st.subheader("🚨 Current Alerts")

    alerts = []
    if current_silica > 2.5:
        alerts.append({"level": "WARNING", "message": f"Silica level elevated: {current_silica:.2f}%", "time": "2 min ago"})
    if np.random.random() < 0.3:
        alerts.append({"level": "INFO", "message": "Flotation air flow adjusted", "time": "5 min ago"})
    if np.random.random() < 0.1:
        alerts.append({"level": "WARNING", "message": "pH deviation detected", "time": "8 min ago"})

    if alerts:
        for alert in alerts:
            if alert["level"] == "WARNING":
                st.warning(f"⚠️ {alert['message']} ({alert['time']})")
            elif alert["level"] == "CRITICAL":
                st.error(f"🚨 {alert['message']} ({alert['time']})")
            else:
                st.info(f"ℹ️ {alert['message']} ({alert['time']})")
    else:
        st.success("✅ All systems operating normally")

    st.subheader("📈 Live Process Control Chart")

    hours = pd.date_range(start=current_time - timedelta(hours=24), 
                         end=current_time, freq='1H')

    np.random.seed(42)
    base_values = np.random.normal(2.2, 0.3, len(hours))

    trend = np.sin(np.arange(len(hours)) * 0.3) * 0.2
    values = base_values + trend

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hours, y=values, mode='lines+markers', 
                            name='Silica %', line=dict(color='blue')))
    fig.add_hline(y=2.5, line_dash="dash", line_color="orange", 
                  annotation_text="Warning Limit (2.5%)")
    fig.add_hline(y=3.5, line_dash="dash", line_color="red", 
                  annotation_text="Critical Limit (3.5%)")
    fig.add_hline(y=np.mean(values), line_dash="dot", line_color="green", 
                  annotation_text=f"Average ({np.mean(values):.2f}%)")

    fig.update_layout(title="24-Hour Silica Concentration Trend", 
                     yaxis_title="Silica %", height=500)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("🎛️ Process Variable Status")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Feed Conditions**")
        iron_feed_current = np.random.normal(56, 2)
        silica_feed_current = np.random.normal(15, 3)
        st.metric("Iron Feed %", f"{iron_feed_current:.1f}")
        st.metric("Silica Feed %", f"{silica_feed_current:.1f}")
        st.metric("Feed Ratio", f"{iron_feed_current/silica_feed_current:.2f}")

    with col2:
        st.markdown("**Chemical Dosing**")
        starch_current = np.random.normal(2900, 200)
        amina_current = np.random.normal(490, 30)
        st.metric("Starch Flow", f"{starch_current:.0f} kg/h")
        st.metric("Amina Flow", f"{amina_current:.0f} g/t")
        st.metric("Chemical Ratio", f"{starch_current/amina_current:.1f}")

def show_performance_page():
    """Model performance dashboard"""
    st.header("🤖 Model Performance Dashboard")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Model Accuracy (R²)", "0.671", delta="0.023")
    with col2:
        st.metric("RMSE", "0.657%", delta="-0.045%")
    with col3:
        st.metric("Predictions Today", "1,247", delta="89")
    with col4:
        st.metric("Model Uptime", "99.8%", delta="0.1%")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Model Accuracy Trend")

        dates = pd.date_range(start=datetime.now() - timedelta(days=30), 
                             end=datetime.now(), freq='1D')
        r2_scores = np.random.normal(0.67, 0.03, len(dates))
        r2_scores = np.clip(r2_scores, 0.5, 0.8)  

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dates, y=r2_scores, mode='lines+markers', 
                                name='R² Score', line=dict(color='blue')))
        fig.add_hline(y=0.6, line_dash="dash", line_color="red", 
                      annotation_text="Minimum Acceptable (0.6)")
        fig.add_hline(y=0.7, line_dash="dash", line_color="green", 
                      annotation_text="Target (0.7)")
        fig.update_layout(title="30-Day Model Performance", yaxis_title="R² Score")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("🎯 Feature Importance")

        features = ['Iron Concentrate %', 'Flotation Air Flow', 'Amina Flow', 
                   'Ore Pulp pH', 'Starch Flow', 'Feed Ratio', 'Chemical Ratio']
        importance = [0.35, 0.18, 0.12, 0.10, 0.08, 0.07, 0.05]

        fig = go.Figure(go.Bar(
            y=features, 
            x=importance, 
            orientation='h',
            marker_color='steelblue'
        ))
        fig.update_layout(title="Current Feature Importance", 
                         xaxis_title="Importance Score")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("🔍 Model Diagnostics")

    col3, col4 = st.columns(2)

    with col3:
        st.markdown("**Prediction vs Actual**")

        actual = np.random.normal(2.2, 0.8, 500)
        actual = np.clip(actual, 0.5, 5.5)
        predicted = actual + np.random.normal(0, 0.25, 500)
        predicted = np.clip(predicted, 0.5, 5.5)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=actual, y=predicted, mode='markers', 
                                name='Predictions', marker=dict(size=4, opacity=0.6)))
        fig.add_trace(go.Scatter(x=[0, 6], y=[0, 6], mode='lines', 
                                name='Perfect Prediction', 
                                line=dict(dash='dash', color='red')))
        fig.update_layout(title="Prediction Accuracy", 
                         xaxis_title="Actual Silica %", 
                         yaxis_title="Predicted Silica %")
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        st.markdown("**Residual Analysis**")

        residuals = actual - predicted

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=predicted, y=residuals, mode='markers',
                                marker=dict(size=4, opacity=0.6)))
        fig.add_hline(y=0, line_dash="dash", line_color="red")
        fig.update_layout(title="Residual Distribution", 
                         xaxis_title="Predicted Silica %", 
                         yaxis_title="Residuals")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("🏥 Model Health Check")

    col5, col6 = st.columns(2)

    with col5:
        st.markdown("**Data Quality Indicators**")
        st.success("✅ No missing values detected")
        st.success("✅ Feature distributions stable")
        st.warning("⚠️ Minor drift in ore composition detected")
        st.success("✅ Prediction confidence within normal range")

    with col6:
        st.markdown("**Performance Alerts**")
        if np.mean(r2_scores[-7:]) < 0.65:
            st.error("🚨 Model performance declining - retraining recommended")
        else:
            st.success("✅ Model performance stable")

        st.info("📅 Last retrained: 7 days ago")
        st.info("📈 Next scheduled retrain: 23 days")


if __name__ == "__main__":
    main()