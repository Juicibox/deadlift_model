import streamlit as st
import cv2
import mediapipe as mp
import joblib
import pandas as pd
import numpy as np


st.set_page_config(page_title="Peso Muerto", page_icon="💪")
st.write("El modelo estima los precios de vivienda para la ciudad de Bogotá, Colombia")
