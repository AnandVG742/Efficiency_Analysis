import streamlit as st
import pandas as pd
import plotly.express as px
import os
from joblib import load

# --- Helper Functions ---
@st.cache_data
def load_data(file_path):
    """
    Loads data from a CSV file.
    """
    if not os.path.exists(file_path):
        st.error(f"Error: The data file '{file_path}' was not found.")
        st.stop()
    df = pd.read_csv(file_path)
    return df

@st.cache_resource
def load_model_and_features(model_path, preprocessor_path):
    """
    Loads the trained model and feature names.
    """
    try:
        model = load(model_path)
        feature_names = load(preprocessor_path)
        return model, feature_names
    except FileNotFoundError:
        st.error("Error: Model files not found. Please run 'train_operational_model.py' first.")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred while loading the model: {e}")
        st.stop()

# --- Main Dashboard Creation Function ---
def create_dashboard(df, model, feature_names):
    """
    Creates the Streamlit dashboard.
    """
    st.set_page_config(layout="wide", page_title="Hospital Operational Dashboard")
    st.title("🏥 Hospital Operational Efficiency Dashboard")
    st.markdown("---")

    # Sidebar for navigation
    st.sidebar.header("Dashboard Navigation")
    tab_selection = st.sidebar.radio("Select a Tab", ["Operational Metrics", "Predict Length of Stay"])

    if tab_selection == "Operational Metrics":
        # Overview Metrics
        st.subheader("Key Operational Metrics")
        col1, col2, col3 = st.columns(3)
        avg_wait_time = df['wait_time_minutes'].mean()
        col1.metric("Average Patient Wait Time", f"{avg_wait_time:.2f} min")
        patients_per_staff = (df['patients_treated'] / df['staff_on_duty']).mean()
        col2.metric("Patients per Staff Member", f"{patients_per_staff:.2f}")
        cost_per_patient = (df['department_cost'] / df['patients_treated']).mean()
        col3.metric("Average Cost per Patient", f"${cost_per_patient:.2f}")

        st.markdown("---")
        st.subheader("Departmental Breakdown")
        
        department_summary = df.groupby('department').agg(
            avg_wait_time=('wait_time_minutes', 'mean'),
            total_patients=('patients_treated', 'sum'),
            total_cost=('department_cost', 'sum')
        ).reset_index()

        department_summary['cost_per_patient'] = department_summary['total_cost'] / department_summary['total_patients']

        # Charts
        st.write("#### Average Wait Time by Department")
        fig1 = px.bar(department_summary, x='department', y='avg_wait_time', color='department',
                      labels={'avg_wait_time': 'Average Wait Time (min)'}, title='Average Wait Time by Department')
        st.plotly_chart(fig1, use_container_width=True)

        st.write("#### Total Patients by Department")
        fig2 = px.pie(department_summary, values='total_patients', names='department',
                      title='Patient Volume by Department')
        st.plotly_chart(fig2, use_container_width=True)

        st.write("#### Cost per Patient by Department")
        fig3 = px.bar(department_summary, x='department', y='cost_per_patient', color='department',
                      labels={'cost_per_patient': 'Cost per Patient ($)'}, title='Cost per Patient by Department')
        st.plotly_chart(fig3, use_container_width=True)
    
    elif tab_selection == "Predict Length of Stay":
        st.subheader("Predict Patient Length of Stay")
        st.markdown("Use the form below to predict how long a patient will stay in a department based on key metrics.")

        # Get unique department names for the dropdown
        departments = df['department'].unique()
        
        # Create a form for user input
        with st.form("prediction_form"):
            selected_department = st.selectbox("Department", options=departments)
            patients_treated = st.number_input("Patients Treated (in a day)", min_value=1, value=15)
            staff_on_duty = st.number_input("Staff on Duty", min_value=1, value=5)

            submitted = st.form_submit_button("Predict")

        if submitted:
            # Create a dataframe from user input, ensuring correct column order
            input_df = pd.DataFrame([[selected_department, staff_on_duty, patients_treated]],
                                    columns=['department', 'staff_on_duty', 'patients_treated'])

            # The model pipeline handles preprocessing
            prediction = model.predict(input_df)[0]
            
            st.markdown("---")
            st.success(f"### Predicted Length of Stay: **{prediction:.2f} days**")
            st.info("This prediction is an estimate based on historical data. Use it as a guide for resource planning.")


if __name__ == '__main__':
    data_file_path = 'operational_data.csv'
    model_file_path = 'operational_model.joblib'
    preprocessor_path = 'feature_names.joblib'
    
    # Check for all required libraries
    try:
        _ = px.bar
        _ = pd.DataFrame
        _ = st.title
        _ = load
    except NameError:
        st.error("Missing required libraries. Please install them using: `pip install streamlit pandas plotly scikit-learn joblib`")
        st.stop()

    try:
        df = load_data(data_file_path)
        model, feature_names = load_model_and_features(model_file_path, preprocessor_path)
        create_dashboard(df, model, feature_names)
    except Exception as e:
        st.error(f"An error occurred during dashboard creation: {e}")
