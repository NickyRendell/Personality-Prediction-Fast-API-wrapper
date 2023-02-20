import pandas as pd
import spacy
import numpy as np
from collections import Counter
from spacy.lang.en import English
from spacy.tokens import Doc
import textacy.resources
from ms import model


nlp = spacy.load('en_core_web_lg')
rs = textacy.resources.DepecheMood(lang="en", word_rep="lemmapos")


def predict(X, model):
    prediction = model.predict(X)
    return prediction

def remove_unwanted_chars(text):
    """"takes out the characters in each line"""
    allowed_chars = """ 0123456789abcdefghijklmnopqrstuvwxyz!()"".,?"""
    clean_text = text.lower()
    for c in clean_text:
        if allowed_chars.find(c) == -1:
            clean_text = clean_text.replace(c, "")
        else:
            pass
    return clean_text



# def get_model_response(input):
#     X = pd.json_normalize(input.__dict__)
#     prediction = predict(X, model)
#     if prediction == 1:
#         label = "M"
#     else:
#         label = "B"
#     return {
#         'label': label,
#         'prediction': int(prediction)
#     }

def get_afraid(text):
    try:
        emot = rs.get_emotional_valence(text)
        return emot['AFRAID'] 
    except:
        return 0

#Get emotional valence of sentence

def get_amused(text):
    try:
        emot = rs.get_emotional_valence(text)
        return emot['AMUSED']
    except:
        return 0

#Get emotional valence of sentence

def get_angry(text):
    try:
        emot = rs.get_emotional_valence(text)
        return emot['ANGRY']
    except:
        return 0

#Get emotional valence of sentence

def get_annoyed(text):
    try:
        emot = rs.get_emotional_valence(text)
        return emot['ANNOYED']
    except:
        return 0
    
#Get emotional valence of sentence

def get_dont_care(text):
    try:
        emot = rs.get_emotional_valence(text)
        return emot['DONT_CARE']
    except:
        return 0
    
#Get emotional valence of sentence

def get_inspired(text):
    try:
        emot = rs.get_emotional_valence(text)
        return emot['INSPIRED']
    except:
        return 0
    
def get_nouns(text):
    c = Counter(([token.pos_ for token in text]))
    total = sum(c.values())
    if total == 0:
        percent = 0
    else:
        percent = (c['NOUN']/total)*100
    return int(percent)

def get_verbs(text):
    c = Counter(([token.pos_ for token in text]))
    total = sum(c.values())
    if total == 0:
        percent = 0
    else:
        percent = (c['VERB']/total)*100
    return int(percent)

def get_adj(text):
    c = Counter(([token.pos_ for token in text]))
    total = sum(c.values())
    if total == 0:
        percent = 0
    else:
        percent = (c['ADJ']/total)*100
    return int(percent)

def get_adv(text):
    c = Counter(([token.pos_ for token in text]))
    total = sum(c.values())
    if total == 0:
        percent = 0
    else:
        percent = (c['ADV']/total)*100
    return int(percent)

def get_intj(text):
    c = Counter(([token.pos_ for token in text]))
    total = sum(c.values())
    if total == 0:
        percent = 0
    else:
        percent = (c['INTJ']/total)*100
    return int(percent)

def get_pronoun(text):
    c = Counter(([token.pos_ for token in text]))
    total = sum(c.values())
    if total == 0:
        percent = 0
    else:
        percent = (c['PRON']/total)*100
    return int(percent)

def get_punct(text):
    c = Counter(([token.pos_ for token in text]))
    total = sum(c.values())
    if total == 0:
        percent = 0
    else:
        percent = (c['PUNCT']/total)*100
    return int(percent)

def get_adp(text):
    c = Counter(([token.pos_ for token in text]))
    total = sum(c.values())
    if total == 0:
        percent = 0
    else:
        percent = (c['ADP']/total)*100
    return int(percent)
    
def vector_panda(doc):
    vect = doc.vector
    return vect   
    

def get_model_response(input):
    d = {'TEXT': [input]}
    text_data = pd.DataFrame(d)
    text_data['TEXT'] = text_data['TEXT'].astype(str)
    text_data['TEXT'] = text_data.TEXT.str.lower()
    text_data['TEXT'] = text_data['TEXT'].apply(remove_unwanted_chars)
    text_data['tokenized'] = text_data['TEXT'].apply(nlp)
    text_data['excl'] = np.where(text_data['TEXT'].str.find('!') >0, 1, 0)
    text_data['quest'] = np.where(text_data['TEXT'].str.find('?') >0, 1, 0)
    text_data['fullst'] = np.where(text_data['TEXT'].str.find('.') >0, 1, 0)
    text_data['comma'] = np.where(text_data['TEXT'].str.find(',') >0, 1, 0)
    text_data['apost'] = np.where(text_data['TEXT'].str.find('"') >0, 1, 0)
    text_data['parenth'] = np.where(text_data['TEXT'].str.find(')') >0, 1, 0)
    text_data['afraid'] = text_data['tokenized'].apply(get_afraid)
    text_data['amused'] = text_data['tokenized'].apply(get_amused)
    text_data['angry'] = text_data['tokenized'].apply(get_angry)
    text_data['annoyed'] = text_data['tokenized'].apply(get_annoyed)
    text_data['dont_care'] = text_data['tokenized'].apply(get_dont_care)
    text_data['inspired'] = text_data['tokenized'].apply(get_inspired)
    text_data['perc_nouns'] = text_data['tokenized'].apply(get_nouns)
    text_data['perc_verbs'] = text_data['tokenized'].apply(get_verbs)
    text_data['perc_adj'] = text_data['tokenized'].apply(get_adj)
    text_data['perc_adv'] = text_data['tokenized'].apply(get_adv)
    text_data['perc_intj'] = text_data['tokenized'].apply(get_intj)
    text_data['perc_pronoun'] = text_data['tokenized'].apply(get_pronoun)
    text_data['perc_punt'] = text_data['tokenized'].apply(get_punct)
    text_data['perc_adp'] = text_data['tokenized'].apply(get_adp)
    text_data['vecs'] = text_data['tokenized'].apply(vector_panda)

    split_df = pd.DataFrame(text_data['vecs'].tolist())

    text_data = pd.concat([text_data, split_df], axis=1)

    text_data.drop(['vecs','tokenized','TEXT'], axis=1, inplace=True)
   

    prediction = predict(text_data, model)

    if prediction == 1:
        label = "High extraversion"
    else:
        label = "Low extraversion"
    return {
        'label': label,
    }