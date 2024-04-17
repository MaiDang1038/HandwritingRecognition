# streamlit run "C:\Users\HP\Downloads\VS CODE\Python_Personal\EndOfTermProject.py" 
import numpy as np
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Activation, Input, Flatten 
from keras.utils import set_random_seed
from keras.backend import clear_session
import time as tm
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff

# User's input
st.title("Handwriting Recognition")
st.write("Draw a letter of your choosing below (A-Z):")
canvas_result = st_canvas(stroke_width=15,
						stroke_color='rgb(255, 255, 255)',
						background_color='rgb(0, 0, 0)',
						height=150,
						width=150,
						key="canvas")

# Prepare X_userinput
if canvas_result.image_data is not None:
    img = canvas_result.image_data
    img = canvas_result.image_data
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_LINEAR)
    X_userinput = img.astype('float32') / 255.0
    X_userinput = X_userinput.reshape(1, 32, 32)

    # Prepare X, y
    ds_path = "C:\\Users\\HP\\Downloads\\Extracted_Images\\Images_Limited" 
    folders = os.listdir(ds_path) 
    X, y = [], []
    with st.expander("Dataset"):
        num_images_per_folder = st.number_input('Number of samples per class',min_value=0,max_value=10000,value=100,step=1)
        on = st.toggle('View Dataset')
        for folder in folders:
            files = os.listdir(os.path.join(ds_path, folder)) 
            png_files = [f for f in files if f.lower().endswith('.png')]
            selected_files = png_files[:num_images_per_folder]
            for f in selected_files:  
                img = Image.open(os.path.join(ds_path, folder, f)) 
                img = np.array(img) 
                X.append(img)
                y.append(folder)
        X = np.array(X)
        y = np.array(y)
        letter_to_number = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'J': 9,
                            'K': 10, 'L': 11, 'M': 12, 'N': 13, 'O': 14, 'P': 15, 'Q': 16, 'R': 17, 'S': 18,
                            'T': 19, 'U': 20, 'V': 21, 'W': 22, 'X': 23, 'Y': 24, 'Z': 25}
        y_num = [letter_to_number[label] for label in y]
        y_num = np.array(y_num)
        st.info(f'Dataset loaded with {num_images_per_folder*26} samples, input shape (32,32), 26 classes', icon="ℹ️")
        if on:
            for letter in letter_to_number.keys():
                folder_path = os.path.join(ds_path, letter)
                files = os.listdir(folder_path)
                png_files = [f for f in files if f.lower().endswith('.png')]
                
                if len(png_files) >= 10:
                    sample_files = np.random.choice(png_files, size=10, replace=False)
                else:
                    sample_files = png_files
                    
                images = [Image.open(os.path.join(folder_path, f)) for f in sample_files]
                st.image(images, width=65)
    col1, col2 = st.columns(2)
    with col1:
        epochs = st.number_input('Epochs',min_value=10,max_value=10000,value=100,step=1)
    with col2:
        test_size = st.number_input('Test size',min_value=0.1,max_value=0.99,value=0.2,step=0.01)
    
# Prepare and train model
if st.button("Train", key=None, type="secondary", use_container_width=True):
    start_time = tm.time()
    # Prepare X_train, X_test, y_train, y_test
    X_train, X_test, y_train, y_test = train_test_split(X, y_num, test_size=test_size, random_state=42)
    y_train_ohe = to_categorical(y_train, num_classes=26)
    y_test_ohe = to_categorical(y_test, num_classes=26)

    #Create and fit model
    clear_session()
    set_random_seed(42)
    np.random.seed(42)
    model = Sequential()
    model.add(Input(shape=X_train.shape[1:]))
    model.add(Flatten())
    model.add(Dense(26, activation='softmax')) 
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='accuracy')
    model.summary()
    history = model.fit(X_train, y_train_ohe, epochs = epochs, verbose=1)
    end_time = tm.time()
    train_time = round(end_time - start_time,0)
    train_loss, train_accuracy = model.evaluate(X_train, y_train_ohe, verbose=0)
    test_loss, test_accuracy = model.evaluate(X_test, y_test_ohe, verbose=0)
    val_loss, val_accuracy = test_loss, test_accuracy
    train_accuracy_rounded = round(train_accuracy*100,0)
    test_accuracy_rounded = round(test_accuracy*100,0)
    st.success(f'Training time: {train_time}s. Train accuracy: {train_accuracy_rounded}%. Test accuracy: {test_accuracy_rounded}%')

    # Draw Loss and Accuracy Graphs
    # history_dict = history.history
    # history_dict['val_loss'] = [val_loss] 
    # history_dict['val_accuracy'] = [val_accuracy] 
    # col3, col4 = st.columns(2)

    # with col3:
    #     train_loss = history_dict['loss']
    #     val_loss = history_dict['val_loss']
    #     graph_data = [train_loss, val_loss]
    #     group_labels = ['Train Loss', 'Val Loss']
    #     fig = ff.create_distplot(
    #         graph_data, group_labels, bin_size=[.3, .3])
    #     st.plotly_chart(fig, use_container_width=True)
    # with col3:
    #     train_accuracy = history_dict['accuracy']
    #     val_loss = history_dict['val_accuracy']
    #     graph_data = [train_accuracy, val_accuracy]
    #     group_labels = ['Train Accuracy', 'Val Accuracy']
    #     fig = ff.create_distplot(
    #         graph_data, group_labels, bin_size=[.3, .3])
    #     st.plotly_chart(fig, use_container_width=True)

    # Prediction
    st.subheader('Prediction:')
    number_to_letter = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
                        10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S',
                        19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'}
    y_pred_prob = model.predict(X_userinput)
    top5_indices = np.argsort(y_pred_prob, axis=1)[0, -5:][::-1]
    top5_letters = [number_to_letter[idx] for idx in top5_indices]
    top5_probabilities = y_pred_prob[0, top5_indices]
    st.write("Top 5 Predictions:")
    for letter, prob in zip(top5_letters, top5_probabilities):
        st.success(f"{letter}: {round(prob*100,2)}%")





