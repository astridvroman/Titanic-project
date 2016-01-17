
# coding: utf-8

# In[1]:

######## IMPORT DE LA TABLE DE DONNEES ###############
from azureml import Workspace

ws = Workspace()
ds = ws.datasets['train.csv']
newframe = ds.to_dataframe()
import pandas


# In[2]:

# suppression des colonnes non Ã©tudiÃ©es
frame = newframe.drop(["PassengerId", "Name", "SibSp", "Parch", "Ticket", "Cabin"], axis=1)


# In[3]:

# traitement des donnÃ©es manquantes de Age et Embarked

# calcul de l'age median
frame["Age"]=frame["Age"].fillna(frame["Age"].median())
# identification de la donnÃ©es la plus reprÃ©sentative dans Embarked
from collections import Counter
list = frame["Embarked"]
print dict(Counter(list))

# S est la modalitÃ© la plus reprÃ©sentative
frame["Embarked"]=frame["Embarked"].fillna("S") 


# In[4]:

# modification du format de PClass et Survived pour que les modalitÃ©s soient reconnues comme quali et non quanti
frame["Pclass"] = pandas.Categorical(frame["Pclass"],ordered=False) # definition de Pclass comme catÃ©gorie pour manipuler les donnÃ©e
frame["Pclass"].cat.rename_categories(["c1", "c2", "c3"]) # conversion des donnÃ©es numÃ©riques en alpha pour faire les indicatrices

frame["Survived"] = pandas.Categorical(frame["Survived"],ordered=False) # definition de Survived comme catÃ©gorie pour manipuler les donnÃ©e
frame["Survived"].cat.rename_categories(["N", "Y"]) # conversion des donnÃ©es numÃ©riques en alpha pour faire les indicatrices


# In[5]:

# DiscrÃ©tiser les variables quantitatives "age" et "fare" en 3 classes de mÃªme taille
frame["AgeQ"]=pandas.qcut(frame.Age,4,labels=["Ag1","Ag2","Ag3", "Ag4"])
frame["FareQ"]=pandas.qcut(frame.Fare,5,labels=["Fa1","Fa2","Fa3", "Fa4", "Fa5"])
frame_temp = frame.drop(["Age", "Fare"], axis=1) # supprimer les variables quanti


# In[6]:

#crÃ©ation d'une table d'indicatrices en utilisation get_dummies sur chacune des data de framefin
frame_temp2=pandas.get_dummies(frame_temp[["Survived","Pclass","Sex", "Embarked", "AgeQ","FareQ"]])
frame_fin = frame_temp2.drop(["Survived_0", "Sex_female"],axis =1) #suppression des variables binaires. conseravtion de l'indicatrice "survivant" et "homme"


# In[7]:

# Extraction des Ã©chantillons d'apprentissage (tables Train) et test (table Test).

# variables explicatives survived_1
B=frame_fin.drop(["Survived_1"],axis=1) # suppression de la variable explicative dans la table de train

# Variable Ã  modÃ©liser
r=frame_fin["Survived_1"] # crÃ©ation d'une table contenant la variable Ã  expliquer

# crÃ©ation des Ã©chantillons de training et de test sur chaque table
import sklearn
from sklearn.cross_validation import train_test_split
B_train,B_test,r_train,r_test=train_test_split(B,r,test_size=0.2,random_state=11) # 20% en test


# In[8]:

from sklearn.linear_model import LogisticRegression
logit = LogisticRegression()
frame_logit=logit.fit(B_train, r_train)


# In[9]:

print "score_test=" 
print frame_logit.score(B_test, r_test)

