import streamlit as st
import pickle
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image

# Page config for better appearance
st.set_page_config(page_title="Sign Language Detector", layout="wide")

# Load the model
model_dict = pickle.load(open('./new-model.p', 'rb'))
model = model_dict['model']

# Initialize Mediapipe and labels
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)
labels_dict = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G'
    
}

# Sidebar for control settings
st.sidebar.title("Control Panel")
if "run" not in st.session_state:
    st.session_state.run = False

def start_prediction():
    st.session_state.run = True

def stop_prediction():
    st.session_state.run = False

st.sidebar.button("Start Prediction", on_click=start_prediction)
st.sidebar.button("Stop Prediction", on_click=stop_prediction)
#confidence_level = st.sidebar.slider("Detection Confidence", 0.1, 1.0, 0.3)

# Display introductory image with caption
col1, col2 = st.columns([1, 2])  # Create two columns to position the image on the left
with col1:
    image = Image.open("American-sign-language-alphabet.png")
    st.image(image, caption="These are the signs!", width=220)

with col2:
    st.write("## Welcome to Sign Language Detection!")
    st.write("Press Start to begin using the sign language detector.")
    st.write("Adjust detection confidence as needed in the sidebar.")

# Placeholder for predicted sign and video frame
predicted_sign = st.empty()
stframe = st.empty()

# Initialize video capture outside the loop to retain state
cap = None

# Main loop for video processing
if st.session_state.run:
    cap = cv2.VideoCapture(0)

    while st.session_state.run and cap.isOpened():
        data_aux = []
        x_ = []
        y_ = []

        ret, frame = cap.read()
        if not ret:
            st.write("Error: Could not read frame.")
            break

        # Convert frame for Mediapipe processing
        H, W, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

                # Bounding box coordinates
                x1, y1 = int(min(x_) * W) - 10, int(min(y_) * H) - 10
                x2, y2 = int(max(x_) * W) - 10, int(max(y_) * H) - 10

                # Make prediction and ensure output is an integer index
                prediction = model.predict([np.asarray(data_aux)])
                predicted_index = int(prediction[0])

                predicted_character = labels_dict.get(predicted_index, "Unknown")

                # Draw bounding box and prediction on the frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                cv2.putText(frame, predicted_character, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

                # Display prediction text in the app
                predicted_sign.markdown(f"<h3 style='text-align: center;'>Predicted Sign: {predicted_character}</h3>", unsafe_allow_html=True)

        # Display frame in Streamlit
        stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

    # Release resources when stopping
    cap.release()
    cv2.destroyAllWindows()

# Footer styling
st.markdown(
    "<style>footer {visibility: hidden;} .footer {text-align: center; font-size: small;}</style>",
    unsafe_allow_html=True
)
st.markdown("<div class='footer'>Built by Vasundhara Sahebrao Aher</div>", unsafe_allow_html=True)
