import streamlit as st
import cv2
import mediapipe as mp
import joblib
import pandas as pd
from landmarks import landmarks
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_tracking_confidence=0.6, min_detection_confidence=0.6)



def main():
    st.set_page_config(page_title="Peso Muerto", page_icon="💪")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        estado_texto = st.subheader('Estado')
        estado_valor = st.text(session_state.current_stage)

    with col2:
        repeticiones_texto = st.subheader('Repeticiones')
        repeticiones_valor = st.text(session_state.counter)

    with col3:
        prob_texto = st.subheader('Prob')
        prob_valor = st.text(session_state.bodylang_prob[session_state.bodylang_prob.argmax()])

    with col4:
        postu_tex = st.subheader('Position')
        postu_valor = st.text(session_state.body_language)

    reset_boton = st.button('Reset', on_click=reset_counter)

    frame_placeholder = st.empty()
    video_capture = cv2.VideoCapture(0)

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        image = detect(frame)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        frame_placeholder.image(image)



        estado_valor.text(session_state.current_stage)
        repeticiones_valor.text(session_state.counter)
        prob_valor.text(session_state.bodylang_prob[session_state.bodylang_prob.argmax()])
        postu_valor.text(session_state.body_language)



    video_capture.release()


if __name__ == "__main__":
    main()
