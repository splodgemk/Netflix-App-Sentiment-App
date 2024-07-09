
## IMPORTS

import streamlit as st
import numpy as np
import pickle
import sklearn
import re 


## Model Loading
with open('sentiment_model.pkl', 'rb') as file:
    rfc = pickle.load(file)

with open('vectorizer_file.pkl', 'rb') as file:
    cv = pickle.load(file)

w = []
c = []

def word_and_char_counts(text):
    words = text.split()
    char_len = 0
    for word in words:
        char_len += len(word) #Character count
    w.append(len(words)) #Word count
    c.append(char_len)
    return (len(words), char_len)

def preprocessor(text):
    text = re.sub("<[^>]*>", "",text)
    emoticons = re.findall("(?::|;|=)(?:-)?(?:\)|\(|D|P)",text)
    text = re.sub("[\W]+", " ", text.lower())+ " ".join(emoticons).replace("-", "")
    return text

def get_sentiment(txt):
    txt = preprocessor(txt)
    word_len, char_len = word_and_char_counts(txt)
    text_features = cv.transform([txt])
    text_features_dense = text_features.toarray()
    char_len_arr = np.array([char_len])
    word_len_arr = np.array([word_len])
    if text_features_dense.ndim > 1:
        text_features_dense = np.squeeze(text_features_dense)
    features = np.hstack((text_features_dense,char_len_arr, word_len_arr))
    features = features.reshape(1, -1)
    print(features.shape)
    prediction = rfc.predict(features)
    if prediction==2:
        return "🙂 Good"
    elif prediction==1:
        return "Neutral"
    return "🙁 Bad"


## Page config
st.set_page_config(
    page_title="Sentiment Analysis Model",
    page_icon = "🤖"
)

with st.sidebar:
    st.header('Sample comments:')
    st.write("Love the app so much. The only thing that annoys me is how they've changed the skip back feature when you're playing video from 10 seconds to 30 seconds. 10 seconds was perfect of you just missed something so i dont know why they changed it to 30, which is way too long. Please change back Netflix!!")
    st.write('')
    st.write("Did have problems but all working fine, I use Netflix most nights as it gets me through long shifts working through the nights in an office so it's a godsend for me, if I was to be critical it would be about the selection of content on offer, I feel every country should have access to the same content, on a trip to Germany I went on Netflix and found many films I'd love to be able to watch but we don't get them in England which is a shame.")
    st.write('')
    st.write("Great that I can watch movies and shows. I got the package with ads, as I don't mind having little break every now and then. It's great that I can watch it directly through app in my TV, just one ad in particular is very annoying, autotrader ad is so loud It literally shakes my windows. It is much louder than series I'm watching and that is very disturbing. For that reason you get 2 stars. Otherwise it would be 5.")

## Main page
st.write("The model is trained on appstore comments for the Netflix app.")
st.header("Sentiment Analysis Bot 🤖")

text = st.text_area("Although any text can be entered for best results copy over appstore comments some sample comments are in the sidebar:",placeholder="Wow this app is pretty cool :)" )

if st.button("Analyse comment"):
    if text:
        #Run the model
        with st.spinner('Running model on cpu...'):
            response = get_sentiment(text)
            
            st.write("Model prediction:")
            st.write(response)
    
    else:
        st.write('Please enter a comment before running the model')
