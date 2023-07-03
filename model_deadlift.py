import streamlit as st
import cv2
import mediapipe as mp
import joblib
import pandas as pd
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_tracking_confidence=0.6, min_detection_confidence=0.6)


st.writer('E')
