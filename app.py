import random
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

import torch_modulation_recognition as tmr


NUM_PER_CLASS = 6000

MODULATIONS = {
    "QPSK": 0,
    "8PSK": 1,
    "AM-DSB": 2,
    "QAM16": 3,
    "GFSK": 4,
    "QAM64": 5,
    "PAM4": 6,
    "CPFSK": 7,
    "BPSK": 8,
    "WBFM": 9,
}

SNRS = [
    0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 
    -20, -18, -16, -14, -12, -10, -8, -6, -4, -2
]

params = {
    "INPUT_TYPE": "random",
    "IDX": 0,
    "BUTTON_PRESSED": False
}

@st.cache(allow_output_mutation=True)
def load_data():
    data = tmr.RadioML2016()
    return data


def get_params(data):

    # st.sidebar.title('Privacy Encoder')
    st.sidebar.title("Options")
    
    # Get signal type
    params["MOD"] = st.sidebar.selectbox("Modulation", list(MODULATIONS.keys()))
    params["SNR"] = st.sidebar.selectbox("Signal-to-Noise Ratio", SNRS)
    
    # Get input type
    options = ["idx", "random"]
    params["INPUT_TYPE"] = st.sidebar.selectbox("Input Options", options)

    if params["INPUT_TYPE"] == "idx":
        params["IDX"] = st.sidebar.slider('idx', min_value=0, max_value=NUM_PER_CLASS, value=params["IDX"])
    elif params["INPUT_TYPE"] == "random":
        params["IDX"] = random.randrange(NUM_PER_CLASS)

    params["BUTTON_PRESSED"] = st.sidebar.button("Click here to plot")

    return params


def main(data):

    params = get_params(data)

    params["SIGNALS"] = data.get_signals(mod=params["MOD"], snr=params["SNR"])

    if params["BUTTON_PRESSED"]:
        fig, ax = plt.subplots(nrows=2, ncols=1, sharey=True, sharex=True)
        plt.xlabel("time (s)")
        fig.suptitle("({}) Modulation: {}, Signal-to-Noise Ratio: {}".format(params["IDX"], params["MOD"], params["SNR"]))
        ax[0].plot(params["SIGNALS"][(params["MOD"], params["SNR"])][params["IDX"], 0, :])
        ax[1].plot(params["SIGNALS"][(params["MOD"], params["SNR"])][params["IDX"], 1, :])
        st.pyplot()

if __name__ == "__main__":
    data = load_data()
    main(data)