# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 08:49:29 2020

@author: Mansour Lo
"""
#Classe Corpus Fournit comme base de travail Ameliorée
#Les specifications des premieres fonctions se trouvent 
#dans les fiches de TD
import datetime as dt
from Author import Author

import pickle
import re
import pandas as pd
import datetime as dt
import numpy as np
import gensim
from statistics import mean
from gensim.parsing.preprocessing import remove_stopwords

class Corpus():
    
    def __init__(self,name):
        self.name = name
        self.collection = {}
        self.authors = {}
        self.id2doc = {}
        self.id2aut = {}
        self.ndoc = 0
        self.naut = 0
            
    def add_doc(self, doc):
        
        self.collection[self.ndoc] = doc
        self.id2doc[self.ndoc] = doc.get_title()
        self.ndoc += 1
        aut_name = doc.get_author()
        aut = self.get_aut2id(aut_name)
        if aut is not None:
            self.authors[aut].add(doc)
        else:
            self.add_aut(aut_name,doc)
            
    def add_aut(self, aut_name,doc):
        
        aut_temp = Author(aut_name)
        aut_temp.add(doc)
        
        self.authors[self.naut] = aut_temp
        self.id2aut[self.naut] = aut_name
        
        self.naut += 1

    def get_aut2id(self, author_name):
        aut2id = {v: k for k, v in self.id2aut.items()}
        heidi = aut2id.get(author_name)
        return heidi

    def get_doc(self, i):
        return self.collection[i]
    
    def get_coll(self):
        return self.collection

    def __str__(self):
        return "Corpus: " + self.name + ", Number of docs: "+ str(self.ndoc)+ ", Number of authors: "+ str(self.naut)
    
    def __repr__(self):
        return self.name

    def sort_title(self,nreturn=None):
        if nreturn is None:
            nreturn = self.ndoc
        return [self.collection[k] for k, v in sorted(self.collection.items(), key=lambda item: item[1].get_title())][:(nreturn)]

    def sort_date(self,nreturn):
        if nreturn is None:
            nreturn = self.ndoc
        return [self.collection[k] for k, v in sorted(self.collection.items(), key=lambda item: item[1].get_date(), reverse=True)][:(nreturn)]
    
    # def save(self,file):
    #         pickle.dump(self, open(file, "wb" ))
            
    
    #la fonction search permet de trouver tous les passages repondants
    #a un pattern REGEX
    def search(self,motcle):
        docs = []
        collection = self.collection
        for i in collection: 
          save = collection[i]
          docs.append(save.get_text()) 
        #Nous concatenons toutes les chaines en une unique chaine
        concat = ". ".join(docs)
        #Nous utilisons une expression reguliere pour recuperer les passages
        result = re.findall(r"(?:[a-zA-Z'-]+[^a-zA-Z'-]+){0,5}%s(?:[^a-zA-Z'-]+[a-zA-Z'-]+){0,5}"%(motcle),concat)
    
        
    #la fonction concorde creer un concordancier sous format Dataframe
    #en utilisant le meme principe que la fonction SEARCH
    def concorde(self,motcle,n):
        docs = []
        collection = self.collection
        for i in collection: 
          save = collection[i]
          docs.append(save.get_text()) 
        #Nous concatenons toutes les chaines en une unique chaine
        concat = ". ".join(docs)
        #Nous utilisons une expression reguliere pour creer le concordancier
        #avec n comme parametre
        result = re.findall(r"(?:[a-zA-Z'-]+[^a-zA-Z'-]+){0,%d}%s(?:[^a-zA-Z'-]+[a-zA-Z'-]+){0,%d}"%(n,motcle,n),concat)
        
        #on cree un dataframe pour stocker ce concordancier
        concord = pd.DataFrame(columns=['contexte gauche', 'motif', 'contexte droit'])
        sepg = list()
        sepd = list()
        motif = list()
        
        #on separe le passage recuperer en deux partie grace au mot cle
        for i in result:
            sep = i.split("{}".format(motcle))
            sepg.append(sep[0])
            motif.append(motcle)
            sepd.append(sep[1])
         #on remplit le concordancier   
        concord['contexte gauche'] = sepg
        concord['motif'] = motif
        concord['contexte droit'] = sepd
     
        
    def nettoyer_texte(self,texte):
        #on met en minuscule le texte
        texte = str.lower(texte)
        #on remplace tous les signes de ponctuations et retour a la ligne
        #et retour chariot par des espaces
        texte = texte.replace('\n', ' ')
        texte = texte.replace('\r', ' ')
        texte = texte.replace(',', ' ')
        texte = texte.replace('.', ' ')
        texte = texte.replace(';', ' ')
        texte = texte.replace('?', ' ')
        texte = texte.replace('!', ' ')
        #permet d'enlever les mots de coordinations pour avoir une analyse
        #la plus realiste possible
        texte = remove_stopwords(texte)
        return texte
    
    
    #Nous definissons le vocabulaire utiliser pour les analyse
    def vocabulaire(self):
        docs = []
        collection = self.collection
        for i in collection: 
          save = collection[i]
          doc = self.nettoyer_texte(save.get_text())
          docs.append(doc) 
        #Nous concatenons toutes les chaines en une unique chaine
        concat = ". ".join(docs)
        #Nous allons creer le tableau freq
        
        #Nous decoupons la chaine en mots
        donnees = concat.split(' ')
        
        #Nous utulisons Pandas Series pour compter les occurances
        #de chaque mot
        donnees = pd.Series(donnees)
        result = pd.DataFrame(donnees.value_counts())
        result.reset_index( inplace=True)
        result.columns = ['mots','frequence']     
        return result
      
     
      #Nous construisons a nouveau un vocabulaire mais en decoupant
      #les documents en n lots distincts
      #Nous renvoyons un Dataframe contenant le nombre d'apparition d'un
      #mot dans une periode de temps données
     
    def vocabulaire_temp(self,n,mot_rechercher)   :
        #Nous definissons des list que l'on utilisera pour pour
        #remplir les Dataframes
        docs = list()
        occ_mot = list()
        
        #Nous recuperons la collection trier par date croissant
        collection = [self.collection[k] for k, v in sorted(self.collection.items(), key=lambda item: item[1].get_date(), reverse=True)]
        
        #Nous calculons le nombre de docs dans chaque partitions
        num_doc_frise = int(self.ndoc/n)
        num_doc_last = self.ndoc - (n-1)*num_doc_frise
        p = 0
        tab = pd.DataFrame()
        date_debut = list()
        date_fin = list()
        
        #Nous creons les partitions et recuperons la date de debut et 
        #de fin de la frise
        for i in range(n):
            
            doc = list()
           
            if i == n-1:
                num = num_doc_last
            else:
                num = num_doc_frise
               
            for j in range(p,p+num):
              if j == p:
                  date_fin.append(collection[j].get_date())
              if j == p+num-1:
                  date_debut.append(collection[j].get_date())
             
             #Nous nettoyons chaque doc avant de l'ajouter a sa partition
              save = collection[j]
              doc1 = self.nettoyer_texte(save.get_text())
              doc.append(doc1)
            #Nous rajoutons la partition aux autres partitions 
            docs.append(doc)  
            p = p+num
        tab['date_fin'] = date_fin
        tab['date_debut'] = date_debut
        
        #Pour chaque partition Nous comptons le nbr d'occurence
        for i in range(n):
            concat = ". ".join(docs[i])
            donnees = concat.split(' ')
            docs[i] = donnees
            
            if mot_rechercher in docs[i]:
                docs[i] = pd.Series(donnees)
                occ_mot.append(docs[i].value_counts()[mot_rechercher])
            else:
                occ_mot.append(0)

        tab['occurence'] = occ_mot   
        return tab
    

    #Cet fonction est l'implementation du score TDIDF en s'aidant
    #la librairie gensim , Nous renvoyons les mots trie par score
    #decroissant
    
    def TFIDF(self):
        #les etapes preliminaires ne changent pas
        docs = []
        docsp = []
        collection = self.collection
        for i in collection: 
          save = collection[i]
          doc = self.nettoyer_texte(save.get_text())
          docs.append(doc) 
        #Nous concatenons toutes les chaines en une unique chaine
        for i in range(len(docs)):
          docsp.append(docs[i].split(' '))
        
        #nous creons un dictionnaire  qui associe un mot a un identifiant
        
        dictionary = gensim.corpora.Dictionary(docsp)
        
        #Nous creons une liste qui contient l'association entre
        # l'identifiant et la frequence d'un mot
        corpus = [dictionary.doc2bow(doc) for doc in docsp]
        
        #Nous appliquons la formule du TDIDF sur chaque mot du corpus      
        tf_idf = gensim.models.TfidfModel(corpus)
        
        #Nous recreons notre vocabulaire
        concat = ". ".join(docs)
        concat = [concat.split(' ')]
        
        #Nous recreons un dictionnaire  qui associe un mot a un identifiant
        dictionary = gensim.corpora.Dictionary(concat)
        
        #Nous rereons le corpus qui va avec
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
    

     #Cet fonction est l'implementation du score d'OKAPI_BM25
     # d'un mot dans chaque document du corpus en s'aidant de
    #la librairie gensim , Nous renvoyons les mots ainsi que sa liste de 
    #score
    def OKAPI_BM25(self):
        #Les etapes preliminaires ne changent pas
        docs = []
        docsplit = []
        liste_mot = []
        liste_score = []
        collection = self.collection
        for i in collection: 
          save = collection[i]
          doc = self.nettoyer_texte(save.get_text())
          docs.append(doc) 
        #Nous concatenons toutes les chaines en une unique chaine
        for i in range(len(docs)):
          docsplit.append(docs[i].split(' '))
        
        #nous creons un dictionnaire  qui associe un mot a un identifiant
        dictionary = gensim.corpora.Dictionary(docsplit)
        
         #Nous creons une liste qui contient l'association entre
        # l'identifiant et la frequence d'un mot
        corpus = [dictionary.doc2bow(doc) for doc in docsplit]
        
        #Nous appliquons la formule du OKAPI_BM25 sur chaque mot 
        #du corpus
        oka_bm25 = gensim.summarization.bm25.BM25(corpus)
        
        #Nous utilisons notre corpus pour calculer ses scores
        concat = ". ".join(docs)
        concat = concat.split(' ')
        
        
        #Nous calculons le score de chaque mot ainsi que sa liste de
        #scores
        for i in range(len(concat)):
          if concat[i] not in liste_mot:
            liste_mot.append(concat[i])
            query_doc = dictionary.doc2bow(concat[i].split())
            scores = oka_bm25.get_scores(query_doc)
            liste_score.append(scores)
        
        #Nous retournons le resultat
        resultat = pd.DataFrame()
        resultat['mots'] = liste_mot
        resultat['score'] = liste_score
        return resultat
        
        

         
