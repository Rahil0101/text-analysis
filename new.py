import streamlit as st
import PyPDF2
import spacy
from nltk.tokenize import word_tokenize
from transformers import BartForConditionalGeneration, BartTokenizer
import textwrap
import re
import spacy
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')

# Load spaCy model for NER and POS
tnlp = spacy.load("en_core_web_sm")

# Function to extract text from PDF
def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text



# Text preprocessing
def preprocess_text(text):
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)

    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)

    # Convert to lowercase
    text = text.lower()

    return text


# summarisation using BART model
def summarisation(text,len_input):
    model_name = "facebook/bart-large-cnn"
    model = BartForConditionalGeneration.from_pretrained(model_name)
    tokenizer = BartTokenizer.from_pretrained(model_name)

    inputs = tokenizer.encode(text, return_tensors="pt",truncation=True, max_length=1024)
    summary_ids = model.generate(inputs, max_length=len_input,min_length = (len_input%100), length_penalty=2.0, num_beams=4,
                                 early_stopping=True)

    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    formatted_summary = "\n".join(textwrap.wrap(summary, width=80))
    return formatted_summary

# Semantic Analysis (sentiment classification) using spaCy
# def analyze_semantics(text):
#     doc = tnlp(text)
#     sentiments = [token.sentiment for token in doc if token.sentiment is not None]
#     return {'sentiment': sum(sentiments) / len(sentiments)} if sentiments else {'sentiment': 0}



# Load spaCy model
nlp = spacy.load("en_core_web_sm")
# Add the textblob component to spaCy pipeline


def analyze_semantics(text):
    doc = nlp(text)
    sentiments = [sent.sentiment for sent in doc.sents if sent.sentiment is not None]
    senti =  sum(sentiments) / len(sentiments) if sentiments else {'sentiment': 0}
    if senti <0:
        st.write('Negative')
    elif senti > 0:
        st.write('Positive')
    else:
        st.write('Neutral')

# NER and POS tagging
def analyze_ner_and_pos(text):
    doc = tnlp(text)
    ner = [(ent.text, ent.label_) for ent in doc.ents]
    pos = [(token.text, token.pos_) for token in doc]
    return ner, pos

# Streamlit app
def main():
    st.set_page_config(layout='wide')
    st.title("Text Analysis Tool")

    # File upload

    uploaded_file = st.file_uploader("Upload a PDF or text file", type=["pdf", "txt"])

    user_input = st.text_area("Input text to be analysed",height=200)

    if uploaded_file or user_input:
        if user_input:
            raw_text = user_input
        elif uploaded_file.type == "application/pdf":
            raw_text = extract_text_from_pdf(uploaded_file)
        else:
            raw_text = uploaded_file.read().decode("utf-8")

        len_input = st.number_input("Input Length of Summary")

        st.subheader("Summary")
        #st.write(raw_text)

        #Preprocess text
        preprocessed_text = preprocess_text(raw_text)

        if len_input>len(preprocessed_text):
            len_input = len(preprocessed_text)

        #Summarisation using BART
        summary = summarisation(preprocessed_text,len_input)
        st.write(summary)

        #Semantic Analysis
        if user_input:
            st.subheader("Semantic Analysis")
            analysis = analyze_semantics(preprocessed_text)
            st.write(analysis)

        #NER and POS
        col_ner, col_pos = st.columns(2, border=True)
        ner, pos = analyze_ner_and_pos(preprocessed_text)
        with col_ner:
            st.header("Named Entity Recognition (NER)")
            for name in ner:
                st.write(name)
        with col_pos:
            st.header("Part-of-Speech (POS) Tagging")
            col1,col2 = st.columns(2)
            for i,p in enumerate(pos):
                if i<(len(pos)/2):
                    with col1:
                        st.write(p)
                else:
                    with col2:
                        st.write(p)

if __name__ == "__main__":
    main()
