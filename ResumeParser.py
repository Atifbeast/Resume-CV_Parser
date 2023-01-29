import streamlit as st
import pickle as pkl
import os
import docx2txt
import spacy
import re
from gensim.models import Word2Vec
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.matcher import PhraseMatcher
import numpy as np
from spacy import displacy

  
nlp = spacy.load('en_core_web_lg')
matcher = PhraseMatcher(nlp.vocab)

st.set_page_config(
    page_title="Resume Scanner",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded",
)

model_word2vec = Word2Vec.load("resumeparsing.model")
model =  pkl.load(open("ResumeModel.pkl", 'rb')) 
encode =  pkl.load(open("ResumeClasses.pkl", 'rb')) 

   
st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 250px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 150px;
        margin-left: -500px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("<h1 style='text-align: centre; color: blue;'>RESUME/CV SCANNER</h1>",
                unsafe_allow_html=True)
st.markdown("<h6 style='text-align: centre; color: white;'>Know where your resume stands :)</h1>",
                unsafe_allow_html=True)


stops = list(STOP_WORDS) 



def extract_text_from_docx(docx_path):
    txt = docx2txt.process(docx_path)
    if txt:
        return txt.replace('\t', ' ')
    return None


def vectorize(list_of_docs, model):
    features = []

    for tokens in list_of_docs:
        zero_vector = np.zeros(model.vector_size)
        vectors = []
        for token in tokens:
            if token in model.wv:
                try:
                    vectors.append(model.wv[token])
                except KeyError:
                    continue
        if vectors:
            vectors = np.asarray(vectors)
            avg_vec = vectors.mean(axis=0)
            features.append(avg_vec)
        else:
            features.append(zero_vector)
    return features


def cleanResume(resumeText):
    resumeText = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",resumeText).split())
    resumeText = re.sub(r'[^\x00-\x7F]+',r' ', resumeText) 
    resumeText = ''.join(resumeText.splitlines())
    return resumeText


with st.sidebar:
    global resume_text, upload
    global resume_text_spacy, re_temp
    upload = st.file_uploader("DRAG AND DROP YOUR RESUME NOW")
    st.markdown("<h5 style='text-align: centre; color: red;'>Only .docx type files accepted</h1>",
                unsafe_allow_html=True)
    if upload:
        try:
            resume_text = extract_text_from_docx(upload)
            resume_text = resume_text.replace('\n\n', ' ')
            re_temp = cleanResume(resume_text)
            resume_text_spacy = nlp(re_temp)
        except Exception:
            st.error('WRONG FILE FORMAT : Only .docx(WORD DOC) type of files are accepted')


scan = st.button('SCAN 📝')
if scan:
    try:
        emails = re.findall(r"[a-z0-9\.\-+_]+@[a-z0-9\.\-+_]+\.[a-z]+", resume_text)
        phone = re.findall(r'[\+\(]?[1-9][0-9 .\-\(\)]{8,}[0-9]', resume_text)
        links = re.findall(r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))", resume_text)
        # for ent in resume_text_spacy.ents:
            # if ent.label_ == "PERSON":
                # st.write("NA"ent.text)

        st.snow()
        # st.write(list(set(emails)))
        if len(list(set(emails))) > 0:
            st.markdown("<h4 style='text-align: centre; color: white;'>EMAIL ✔️ </h1>",
                unsafe_allow_html=True)
            st.success(list(set(emails)))
        else:
            st.markdown("<h4 style='text-align: centre; color: white;'>EMAIL ❌ </h1>",
                unsafe_allow_html=True)
            st.error('Email-Id is not present try including it in your Resume')



        if len(list(set(phone))) > 0:
            st.markdown("<h4 style='text-align: centre; color: white;'>MOBILE NO ✔️ </h1>",
                unsafe_allow_html=True)
            st.success(list(set(phone)))
        else:
            st.markdown("<h4 style='text-align: centre; color: white;'>MOBILE NO ❌ </h1>",
                unsafe_allow_html=True)
            st.error('Mobile number is not present try including it in your Resume')


        
        if len(list(set(links))) > 0:
            st.markdown("<h4 style='text-align: centre; color: white;'>LINKS ✔️ </h1>",
                unsafe_allow_html=True)
            st.success(list(set(links)))
        else:
            st.markdown("<h4 style='text-align: centre; color: white;'>LINKS ❌</h1>",
                unsafe_allow_html=True)
            st.error("Link's are not present try including your Github or LinkedIn Profile in your Resume")

        with st.spinner('Analyzing on it...Plz Wait...'):

            d = [] 
            for i in encode.classes_:
                d.append(i)

            with open("linkedin_skill.txt", 'r', encoding='utf-8') as file:
                txt = file.read()

            txt = txt.split('\n')
            ev = [nlp.make_doc(i) for i in txt]
            matcher.add("SKILLS", None, *ev)
            get_skills = matcher(resume_text_spacy)

            demo = []
            for match_id, start, end in get_skills:
                span = resume_text_spacy[start : end]
                demo.append(span.text)

            re_text = ' '.join(demo)  
            my_skills_re_text = re_text
            my_skills_clean_re_text = cleanResume(my_skills_re_text)

            nlp_one = nlp(my_skills_clean_re_text)
            nlp_one_lst = []
            for i in nlp_one:
                if i.text.lower() not in nlp_one_lst:
                    nlp_one_lst.append(i.text.lower())
            
            nlp_one_lst = [nlp_one_lst]
            vecs2 = vectorize(nlp_one_lst, model_word2vec)
            pred2 = model.predict(vecs2) 
            temp2 = []

            # st.write(nlp_one_lst)

            for i in pred2:
                for j in i:
                    temp2.append(j)

            probs2 = []
            for i in model.predict_proba(vecs2):
                for j in i:
                    probs2.append(j)

            for (i, j, k) in zip(d, temp2, probs2):
                if j == 1:
                    st.write(' ')
                    st.write(' ')
                    st.write(' ')
                    st.header(f'YOUR SKILLS MATCHES "{k*100}%" FOR "{i.upper()}"')

    except Exception as e:
        st.error("This error wasn't suppose to happen😲 try uploading your file again")

