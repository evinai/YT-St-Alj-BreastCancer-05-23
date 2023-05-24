import streamlit as st
import pickle5 as pickle
import pandas as pd
import plotly.graph_objects as go
import numpy as np

data_path = '../data/data.csv'


def get_clean_data(bc_data):
    """
    Reads in the data from the specified path and returns a pandas dataframe.
    """
    df = pd.read_csv(bc_data)

    df = df.drop(columns=['diagnosis', 'Unnamed: 32', 'id'], axis=1)

    return df


def get_scaled_values(df, input_data):
    scaled_values = {}
    for column in df.columns:
        scaled_values[column] = (input_data[column] - df[column].min()) / (df[column].max() - df[column].min())
    return scaled_values


def create_sliders(df):
    input_dict = {}
    for column in df.columns:
        input_dict[column] = st.slider(column,
                                       min_value=float(0),
                                       max_value=float(df[column].max()),
                                       value=float(df[column].mean())
                                       )
    return input_dict


def get_radar_chart(input_data):
    df = get_clean_data(data_path)
    input_data = get_scaled_values(df, input_data)
    categories = [i.title().replace('_Mean', '') for i in df.columns]

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=[v for i, v in input_data.items()][:9],
        theta=categories,
        fill='toself',
        name="Mean Value"
    ))
    fig.add_trace(go.Scatterpolar(
        r=[v for i, v in input_data.items()][10:19],
        theta=categories,
        fill='toself',
        name='Standard Error'
    ))

    fig.add_trace(go.Scatterpolar(
        r=[v for i, v in input_data.items()][20:],
        theta=categories,
        fill='toself',
        name='Standard Error'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True
    )

    return fig


def add_predictions(input_data):
    model = pickle.load(open("../model/model.pkl", "rb"))
    scaler = pickle.load(open("../model/scaler.pkl", "rb"))

    input_array = np.array(list(input_data.values())).reshape(1, -1)

    input_array_scaled = scaler.transform(input_array)

    prediction = model.predict(input_array_scaled)

    st.subheader("Cell cluster prediction")
    st.write("The cell cluster is:")

    if prediction[0] == 0:
        st.write("<span class='diagnosis benign'>Benign</span>", unsafe_allow_html=True)
    else:
        st.write("<span class='diagnosis malicious'>Malicious</span>", unsafe_allow_html=True)

    st.write("Probability of being benign: ", model.predict_proba(input_array_scaled)[0][0])
    st.write("Probability of being malicious: ", model.predict_proba(input_array_scaled)[0][1])

    st.write(
        "This app can assist medical professionals in making a diagnosis, but should not be used as a substitute for a professional diagnosis.")


def home_page():
    st.title("Home")
    st.write("This is a simple web app to predict whether "
             "a patient has breast cancer or not.")
    st.markdown("##")


def main():
    st.set_page_config(
        page_title="Breast Cancer Predictor",
        page_icon=":female-doctor:",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    with open("style.css") as f:
        test = st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)

    st.markdown(test)

    with st.sidebar:
        st.subheader("Cell Nuclei Measurements")
        df = get_clean_data(data_path)
        input_data = create_sliders(df)

    with st.container():
        st.title("Breast Cancer Predictor")
        st.write(
            "Please connect this app to your cytology lab to help diagnose breast cancer form your tissue sample. This app predicts using a machine learning model whether a breast mass is benign or malignant based on the measurements it receives from your cytosis lab. You can also update the measurements by hand using the sliders in the sidebar. ")

    col1, col2 = st.columns([5, 2])

    with col1:
        radar_chart = get_radar_chart(input_data)
        st.plotly_chart(radar_chart)
    with col2:
        add_predictions(input_data)


if __name__ == '__main__':
    main()
