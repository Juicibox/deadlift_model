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


class SessionState:
    def __init__(self):
        self.current_stage = ''
        self.counter = 0
        self.bodylang_prob = np.array([0, 0])
        self.bodylang_class = ''
        self.body_language = ''


# new instance SessionState
session_state = SessionState()

def detect(frame):

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(106, 13, 173), thickness=4, circle_radius=5),
                              mp_drawing.DrawingSpec(color=(255, 102, 0), thickness=5, circle_radius=10))
    try:
        row = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten().tolist()
        X = pd.DataFrame([row], columns = landmarks)
        session_state.bodylang_prob  = model.predict_proba(X)[0]
        session_state.bodylang_class = model.predict(X)[0]
        session_state.body_language = model1.predict(X)[0]

        if session_state.bodylang_class== "down" and session_state.bodylang_prob[session_state.bodylang_prob.argmax()] > 0.7:
            session_state.current_stage = "down"
        elif session_state.current_stage == "down" and session_state.bodylang_class == "up" and session_state.bodylang_prob[session_state.bodylang_prob.argmax()] > 0.7:
            session_state.current_stage = "up"
            session_state.counter += 1

    except Exception as e:
        print(e)

    return image
def reset_counter():

    session_state.counter = 0


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
