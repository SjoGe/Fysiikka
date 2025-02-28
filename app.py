import streamlit as st
import pandas as pd
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
from geopy.distance import geodesic
import folium
from streamlit_folium import st_folium

# Ladataan tiedostot
st.title("Kävelyn Analyysi")
accel_file = st.file_uploader("Lataa Linear Accelerometer.csv", type=["csv"])
gps_file = st.file_uploader("Lataa Location.csv", type=["csv"])

if accel_file and gps_file:
    # Ladataan kiihtyvyysdata
    accel_data = pd.read_csv(accel_file)
    accel_data["Magnitude"] = np.sqrt(accel_data["X (m/s^2)"]**2 + accel_data["Y (m/s^2)"]**2 + accel_data["Z (m/s^2)"]**2)
    
    # Suodatetaan kiihtyvyysdata
    fs = 1 / np.mean(np.diff(accel_data["Time (s)"]))  # Näytteenottotaajuus
    lowcut, highcut = 0.8, 3.0
    b, a = signal.butter(2, [lowcut / (fs / 2), highcut / (fs / 2)], btype='band')
    accel_data["Filtered Magnitude"] = signal.filtfilt(b, a, accel_data["Magnitude"])
    
    # Lasketaan askelmäärä
    peaks, _ = signal.find_peaks(accel_data["Filtered Magnitude"], height=0.5, distance=fs/2)
    step_count_filtered = len(peaks)
    
    # Fourier-analyysi
    fft_vals = np.fft.rfft(accel_data["Filtered Magnitude"])
    fft_freqs = np.fft.rfftfreq(len(accel_data), d=1/fs)
    peak_freq_index = np.argmax(np.abs(fft_vals[1:])) + 1
    step_frequency = fft_freqs[peak_freq_index]
    duration = accel_data["Time (s)"].iloc[-1] - accel_data["Time (s)"].iloc[0]
    step_count_fourier = int(step_frequency * duration)
    
    # Ladataan GPS-data
    gps_data = pd.read_csv(gps_file)
    mean_velocity = gps_data["Velocity (m/s)"].mean()
    
    distance = 0
    for i in range(1, len(gps_data)):
        coord1 = (gps_data.loc[i-1, "Latitude (°)"], gps_data.loc[i-1, "Longitude (°)"])
        coord2 = (gps_data.loc[i, "Latitude (°)"], gps_data.loc[i, "Longitude (°)"])
        distance += geodesic(coord1, coord2).meters
    
    step_length = distance / step_count_filtered

    
    # Tulostetaan laskelmat
    st.subheader("Analyysin tulokset")
    st.write(f"**Askelmäärä (suodatettu kiihtyvyysdata):** {step_count_filtered}")
    st.write(f"**Askelmäärä (Fourier-analyysi):** {step_count_fourier}")
    st.write(f"**Keskinopeus:** {mean_velocity:.2f} m/s")
    st.write(f"**Kuljettu matka:** {distance:.2f} metriä")
    st.write(f"**Askelpituus:** {step_length:.2f} metriä")
    
    # Piirretään kuvaajat
    st.subheader("Visualisoinnit")
    fig1, ax1 = plt.subplots()
    ax1.plot(accel_data["Time (s)"], accel_data["Filtered Magnitude"], label="Suodatettu kiihtyvyys")
    ax1.set_xlabel("Aika (s)")
    ax1.set_ylabel("Kiihtyvyys (m/s²)")
    ax1.set_title("Suodatettu kiihtyvyysdata")
    st.pyplot(fig1)
    
    fig2, ax2 = plt.subplots()
    ax2.plot(fft_freqs, np.abs(fft_vals), label="Tehospektritiheys")
    ax2.axvline(step_frequency, color='r', linestyle='--', label=f"Askelfrekvenssi: {step_frequency:.2f} Hz")
    ax2.set_xlabel("Taajuus (Hz)")
    ax2.set_ylabel("Amplitude")
    ax2.set_title("Fourier-analyysi")
    st.pyplot(fig2)

        # Karttanäkymä GPS-datasta
    st.subheader("Reitti kartalla (interaktiivinen)")

    # Hae reitin keskipiste
    center_lat = gps_data["Latitude (°)"].mean()
    center_lon = gps_data["Longitude (°)"].mean()

    # Luo kartta
    m = folium.Map(location=[center_lat, center_lon], zoom_start=15)

    # Lisää reitti kartalle
    route = list(zip(gps_data["Latitude (°)"], gps_data["Longitude (°)"]))
    folium.PolyLine(route, color="blue", weight=5, opacity=0.7).add_to(m)

    # Näytä kartta Streamlitissä
    st_folium(m, width=700, height=500)
    

st.write("**Lataa CSV-tiedostot yllä ja näe analyysi ja kuvaajat!**")
