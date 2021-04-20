#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Wed Apr  1 14:49:13 2020

@author: julien et antoine 
"""

# package permettant d'incrémenter l'identifiant unique à attribuer à un document
#import itertools

#
# classe mère permettant de modéliser un Document (au sens large)
#

#Classe Document Fournit comme base de travail
#Les specifications de chaque fonction se trouvent dans les fiches de TD
import datetime as dt
#from gensim.summarization.summarizer import summarize
#from gensim import summarize

class Document():
    
    # constructor
    def __init__(self, date, title, author, text, url):
        self.date = date
        self.title = title
        self.author = author
        self.text = text
        self.url = url
    
    # getters
    
    def get_author(self):
        return self.author

    def get_title(self):
        return self.title
    
    def get_date(self):
        return self.date
    
    def get_source(self):
        return self.source
   
    def get_text(self):
        return self.text

    def __str__(self):
        return "Document " + self.getType() + " : " + self.title
    
    def __repr__(self):
        return self.title

    def sumup(self,ratio):
        try:
            auto_sum = summarize(self.text,ratio=ratio,split=True)
            out = " ".join(auto_sum)
        except:
            out =self.title            
        return out
    
    def getType(self):
        pass
    
# classe fille permettant de modéliser un Document Reddit
#

class RedditDocument(Document):
    
    def __init__(self, date, title,
                 author, text, url, num_comments):        
        Document.__init__(self, date, title, author, text, url)
        # ou : super(...)
        self.num_comments = num_comments
        self.source = "Reddit"
        
    def get_num_comments(self):
        return self.num_comments

    def getType(self):
        return "reddit"
    
    def __str__(self):
        #return(super().__str__(self) + " [" + self.num_comments + " commentaires]")
        return Document.__str__(self) + " [" + str(self.num_comments) + " commentaires]"
    
#
# classe fille permettant de modéliser un Document Arxiv
#

class ArxivDocument(Document):
    
    def __init__(self, date, title, author, text, url, coauteurs):
        #datet = dt.datetime.strptime(date, '%Y-%m-%dT%H:%M:%SZ')
        Document.__init__(self, date, title, author, text, url)
        self.coauteurs = coauteurs
    
    def get_num_coauteurs(self):
        if self.coauteurs is None:
            return(0)
        return(len(self.coauteurs) - 1)

    def get_coauteurs(self):
        if self.coauteurs is None:
            return([])
        return(self.coauteurs)
        
    def getType(self):
        return "arxiv"

    def __str__(self):
        s = Document.__str__(self)
        if self.get_num_coauteurs() > 0:
            return s + " [" + str(self.get_num_coauteurs()) + " co-auteurs]"
        return s
    
