import sys
import random
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

import torch_modulation_recognition as tmr


NUM_PER_CLASS = 6000

params = {
    "INPUT_TYPE": "random",
    "IDX": 0,
    "BUTTON_PRESSED": False
}

@st.cache(allow_output_mutation=True)
def load_data():
    data = tmr.data.RadioML2016()
    return data

def get_params(data):

    # st.sidebar.title('Privacy Encoder')
    st.sidebar.title("Options")
    
    # Get signal type
    params["MOD"] = st.sidebar.selectbox("Modulation", list(tmr.data.MODULATIONS.keys()))
    params["SNR"] = st.sidebar.selectbox("Signal-to-Noise Ratio", tmr.data.SNRS)
    
    # Get input type
    options = ["idx", "random"]
    params["INPUT_TYPE"] = st.sidebar.selectbox("Input Options", options)

    if params["INPUT_TYPE"] == "idx":
        params["IDX"] = st.sidebar.slider(
            'idx',
            min_value=0,
            max_value=NUM_PER_CLASS,
            value=params["IDX"]
        )
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
        fig.suptitle("({}) Modulation: {}, Signal-to-Noise Ratio: {} (In-Phase (I) and Quadrature (Q) Channels)".format(params["IDX"], params["MOD"], params["SNR"]))
        ax[0].plot(params["SIGNALS"][(params["MOD"], params["SNR"])][params["IDX"], 0, :])
        ax[1].plot(params["SIGNALS"][(params["MOD"], params["SNR"])][params["IDX"], 1, :])
        st.pyplot()

if __name__ == "__main__":
    data = load_data()
    main(data)