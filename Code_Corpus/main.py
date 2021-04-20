# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 18:43:14 2021

@author: Mansour Lo
"""
import pickle
import re
import Document
import Author
import Corpus
from gensim.parsing.preprocessing import remove_stopwords
from gensim.utils import tokenize
from gensim.test.utils import common_texts
from gensim.models import Word2Vec
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import gensim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os
import unicodedata,contractions
from bs4 import BeautifulSoup

import seaborn as sns
sns.set_style("darkgrid")

from sklearn.decomposition import PCA

os.chdir("C:/Users/ashamza/Google Drive/Lyon 2/Initiation à la recherche/initRech\initRech")
file = open('Bigdata.crp', 'rb')

corpus = pickle.load(file)


file.close()

#concat documents
def concat_resume(corpus):
       docs = []
       collection = corpus.get_coll()
       for i in collection: 
          save = collection[i]
          docs.append(save.get_text()) 
        #Nous concatenons toutes les chaines en une unique chaine
       concat = ". ".join(docs)
       return concat
   
#Nombre de mots avant le nettoyage 
concatenation = concat_resume(corpus)
count_word = concatenation.split(' ') #
print (len(count_word)) #84457

#nettoyage du texte
def nettoyer_texte(texte):
        texte = BeautifulSoup(texte)
        texte = texte.get_text()
        #on met en minuscule le texte
        texte = str.lower(texte)
        #on remplace tous les signes de ponctuations et retour a la ligne
        #et retour chariot par des espaces
        texte = texte.replace('\n', ' ')
        texte = texte.replace('\r', ' ')
        
        texte = re.sub('\[[^]]*\]', '', texte)
        texte = contractions.fix(texte)
        texte = re.sub("[^a-zA-Z]", " ",texte); 
        pattern = r"((?<=^)|(?<= )).((?=$)|(?= ))"
        texte = re.sub("\s+", " ", re.sub(pattern, '', texte).strip())
        return texte

#Nombre des mots après nettoyage
texte_nettoyer = nettoyer_texte(concatenation)
count_after_clean = texte_nettoyer.split(' ')
print(len(count_after_clean))#77646


#fonction de stopword
def remove_stopword(texte):
    texte = remove_stopwords(texte)
    return texte

#Nombre des mots après la suppression des Stop-words
texte_wth_stpword = remove_stopwords(texte_nettoyer)
donnees2 = texte_wth_stpword.split(' ')
print(len(donnees2))#47305

#Fonction de tokenization
def texte_to_token(texte):
     tokens = list(tokenize(texte))
     new_words = []
     for word in tokens:
        new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        new_words.append(new_word)
     return new_words
 
tokens = texte_to_token(texte_wth_stpword)

#Fonction de stemming 
def racinisation_texte(tokens):
    porter = PorterStemmer()
    stem=[]
    for word in tokens:
        stem.append(porter.stem(word))
    return stem

racine = racinisation_texte(tokens)

#Fonction de lemmatisation
def lemmatisation_texte(tokens):
    lemmatizer=WordNetLemmatizer()
    lem=[]
    for word in tokens:
        lem.append(lemmatizer.lemmatize(word))
    return lem

lemmatization = lemmatisation_texte(tokens)

#Définition de la fonction TF-IDF
def TF_IDF(corpus,methode): #methode = lemmatisation ou racinisation
    
        docs = []
        docsp = []
        collection = corpus.get_coll()
        for i in collection: 
          save = collection[i]
          doc = nettoyer_texte(save.get_text())
          doc = remove_stopword(doc)
          docs.append(doc) 
        #Nous concatenons toutes les chaines en une unique chaine
        for i in range(len(docs)):
          docsp.append(methode(texte_to_token(docs[i])))
        
        #nous creons un dictionnaire  qui associe un mot a un identifiant
        
        dictionary = gensim.corpora.Dictionary(docsp)
        
        #Nous creons une liste qui contient l'association entre
        # l'identifiant et la frequence d'un mot
        corpus = [dictionary.doc2bow(doc) for doc in docsp]
        
        #Nous appliquons la formule du TDIDF sur chaque mot du corpus      
        tf_idf = gensim.models.TfidfModel(corpus)
        
        #Nous recreons notre vocabulaire
        concat = ""
        for i in range(len(docsp)): 
          concat = concat + " ".join(docsp[i])
        concat = [concat.split(' ')]
        
        #Nous recreons un dictionnaire  qui associe un mot a un identifiant
        dictionary = gensim.corpora.Dictionary(concat)
        
        #Nous recreons le corpus qui va avec
        corpus = [dictionary.doc2bow(doc) for doc in concat]
        
        
        #Pour chaque document de notre corpus ici il y'en qu'un
        #nous calculons le score de chaque mot
        for doc in tf_idf[corpus]:
            score = [[dictionary[id], np.around(freq, decimals=2)] for id, freq in doc]
        
        #Puis nous recuperons et renvoyons le resultat
        resultat = pd.DataFrame()
        liste_mot = [score[i][0] for i in range(len(score))]
        liste_score = [score[i][1] for i in range(len(score))]
        resultat['mots'] = liste_mot
        resultat['score'] = liste_score
        resultat = resultat.sort_values(by ='score',ascending=False)
        return resultat

tf_idf_lem = TF_IDF(corpus,lemmatisation_texte)
print(len(tf_idf_lem.mots)) # 5820 mots après lemmatisation puis TF-IDF


tf_idf_rac = TF_IDF(corpus,racinisation_texte)
print(len(tf_idf_rac.mots)) # 4094 mots après racinisation puis TF-IDF


# cette fonction permet de supprimer les mots non utiles(fréquence=0)  après TF-IDF  
def treeshold_TF_IDF(dataframe):
    data = dataframe.loc[dataframe['score']>0]
    return data

mot_freq_tfidf_lem = treeshold_TF_IDF(tf_idf_lem)
print(len(mot_freq_tfidf_lem)) # 529 mots après suppression des mots avec freq=0

mot_freq_tfidf_rac = treeshold_TF_IDF(tf_idf_rac)
print(len(mot_freq_tfidf_rac)) # 465 mots après suppression des mots avec freq=0
 
   
def doc_tokenization(corpus,methode):
    
        docs = []
        docsp = []
        collection = corpus.get_coll()
        for i in collection: 
          save = collection[i]
          doc = nettoyer_texte(save.get_text())
          doc = remove_stopword(doc)
          docs.append(doc) 
        #Nous concatenons toutes les chaines en une unique chaine
        for i in range(len(docs)):
          docsp.append(methode(texte_to_token(docs[i])))
          
        return docsp

#les tokens de chaque corpus avec une méthode = (lemmatisation/racinisation)
doc_token_lem = doc_tokenization(corpus,lemmatisation_texte)
doc_token_rac = doc_tokenization(corpus,racinisation_texte)

#Définition de la fonction word2vec 
def word2vec(listtoken):
    model = Word2Vec(listtoken, vector_size=100, window=30,
                          min_count=10,workers=1,epochs=50)
    return model

# word2vec  

word2vec_token_lem = word2vec(doc_token_lem)
print(len(word2vec_token_lem.wv.index_to_key)) # 992 mots

word2vec_token_rac = word2vec(doc_token_rac)
print(len(word2vec_token_rac.wv.index_to_key)) # 921 mots


def tfidf_vect(list_tfidf,model):
    mat = pd.DataFrame()
    mat["mots"] = list_tfidf
    
    vecteur = []
    
    for word in list_tfidf:
        vecteur.append(model.wv[word])
        
    mat["vecteur"] = vecteur
    
    return mat

def tsne_plot(model, word): 
    arrays = np.empty((0, 100), dtype='f')
    word_labels = [word]
    color_list  = ['red']

    # adds the vector of the query word
    arrays = np.append(arrays, model.wv.__getitem__([word]), axis=0)
    
    # gets list of most similar words
    close_words = model.wv.most_similar([word])
    
    # adds the vector for each of the closest words to the array
    for wrd_score in close_words:
        wrd_vector = model.wv.__getitem__([wrd_score[0]])
        word_labels.append(wrd_score[0])
        color_list.append('blue')
        arrays = np.append(arrays, wrd_vector, axis=0)
           
    # Reduces the dimensionality from 300 to 50 dimensions with PCA
    reduc = PCA(n_components=2).fit_transform(arrays)
    
    # Finds t-SNE coordinates for 2 dimensions
    np.set_printoptions(suppress=True)
    
    Y = TSNE(n_components=2, random_state=0, perplexity=15).fit_transform(reduc)
    
    # Sets everything up to plot
    df = pd.DataFrame({'x': [x for x in Y[:, 0]],
                       'y': [y for y in Y[:, 1]],
                       'words': word_labels,
                       'color': color_list})
    
    fig, _ = plt.subplots()
    fig.set_size_inches(9, 9)
    
    # Basic plot
    p1 = sns.regplot(data=df,
                     x="x",
                     y="y",
                     fit_reg=False,
                     marker="o",
                     scatter_kws={'s': 40,
                                  'facecolors': df['color']
                                 }
                    )
    
    # Adds annotations one by one with a loop
    for line in range(0, df.shape[0]):
         p1.text(df["x"][line],
                 df['y'][line],
                 '  ' + df["words"][line].title(),
                 horizontalalignment='left',
                 verticalalignment='bottom', size='medium',
                 color=df['color'][line],
                 weight='normal'
                ).set_size(15)

    
    plt.xlim(Y[:, 0].min()-50, Y[:, 0].max()+50)
    plt.ylim(Y[:, 1].min()-50, Y[:, 1].max()+50)
            
    plt.title('t-SNE visualization for {}'.format(word.title()))
          

test = TF_IDF(corpus,lemmatisation_texte)
test2 = treeshold_TF_IDF(test)

vocab_tfidf = test2["mots"].tolist()

test3 = doc_tokenization(corpus,lemmatisation_texte)
final = word2vec(test3)  
 
mat = tfidf_vect(vocab_tfidf, final)

tsne_plot(final, 'data')

dic={'mots totals':84457,
     'Aprés nettoyage': 77646,
     'Après Stopword':47305,
     'lemmatisation/TF-IDF':5820,
     'racinisation/TF-IDF':4094
     }
#affichage plot
df=pd.DataFrame.from_dict(dic,orient='index')
df.plot(kind='bar',title="évolution du prétraitement")

dic2={
      'Sup_freq_lem': 529,
     'Sup_freq_rac': 465}

df=pd.DataFrame.from_dict(dic2,orient='index')
df.plot(kind='bar',title="Après éffacement des mots non fréquents")


