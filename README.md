# Efficiency_Analysis
# 🏥 Hospital Operational Efficiency Dashboard

## Project Description

This project provides a comprehensive Streamlit dashboard for analyzing and predicting key operational metrics in a hospital setting. The dashboard visualizes historical data on various hospital departments and uses a machine learning model to predict a patient's **length of stay**.

The application consists of two main parts:
1.  **Data Analysis:** Interactive charts to explore trends in patient volume, staff efficiency, and wait times across different departments.
2.  **Predictive Tool:** A user-friendly form that allows hospital staff to input a patient's information and get an immediate prediction of their length of stay, aiding in resource planning and bed management.

## Project Files

* `HROE_Analysis.py`: The main Streamlit application script.
* `train_operational_model.py`: A Python script used to train the machine learning model and save it.
* `operational_data.csv`: The dataset containing historical operational data.
* `operational_model.joblib`: The pre-trained machine learning model.
* `feature_names.joblib`: A file containing the list of feature names used by the model.

## How to Run the App Locally

To set up and run this project on your local machine, follow these steps:

1.  **Clone the Repository:**
    ```bash
    git clone <your_repository_url>
    cd <your_repository_name>
    ```

2.  **Create and Activate a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    # On Windows:
    venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```

3.  **Install Required Libraries:**
    Install all the necessary packages using the `requirements.txt` file provided.
    ```bash
    pip install -r requirements.txt
    ```

4.  **Train the Model:**
    The dashboard requires the pre-trained model files. Run the training script once to generate them.
    ```bash
    python train_operational_model.py
    ```
    This will create `operational_model.joblib` and `feature_names.joblib` in your project directory.

5.  **Run the Streamlit App:**
    ```bash
    streamlit run HROE_Analysis.py
    ```

The app will open in your web browser.

## Deployment

This app can be deployed to cloud platforms like Streamlit Cloud, Hugging Face Spaces, or Render by connecting it to your GitHub repository.
