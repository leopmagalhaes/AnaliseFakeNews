#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 11:14:20 2020

@author: usuario
"""

from textblob.classifiers import NaiveBayesClassifier
import pandas as pd

news = pd.read_csv('news.csv', sep=';', header=1)
clf = NaiveBayesClassifier(news.values, format="csv")
frase = input("Digite a frase a ser verificada: ")
classificacao = clf.classify(frase)
dist_prob = clf.prob_classify(frase)
print("A frase e:", classificacao)
fake = dist_prob.prob('fake')
print("Esta frase tem", fake*100, "% de chances de ser fake")
verdadeiro = dist_prob.prob('verdadeiro')
print("Esta frase tem", verdadeiro*100, "% de chances de ser verdadeira")
#clf.show_informative_features()