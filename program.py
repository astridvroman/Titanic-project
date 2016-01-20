
# coding: utf-8

# In[54]:

######## IMPORT DE LA TABLE DE DONNEES ###############
from azureml import Workspace

ws = Workspace(
    workspace_id='5fea9ff155714fa59da6bc15e9cfddc0',
    authorization_token='e1ed5eab2ac040c4a1c8ed5ad14621fb',
    endpoint='https://studioapi.azureml.net'
)
ds = ws.datasets['train.csv']
frame = ds.to_dataframe()

dt= ws.datasets['test.csv']
titanic = dt.to_dataframe()
import pandas


# In[60]:

def traitement(frame,with_surv_column) :# Preparation de la table - traitement des donnÃ©es
    
    newframe=frame.copy() #copie de la table d'orgine pour ne pas l'Ã©craser
    
    #### TRAITEMENT des donnÃ©es manquantes de Age et Embarked
    # calcul de l'age median
    newframe["Age"]=newframe["Age"].fillna(newframe["Age"].median())
    # identification de la donnÃ©es la plus reprÃ©sentative dans Embarked
    from collections import Counter
    list = newframe["Embarked"]
    print dict(Counter(list))    # S est la modalitÃ© la plus reprÃ©sentative
    # remplacement de la donnÃ©e manquante par S
    newframe["Embarked"]=newframe["Embarked"].fillna("S") 

    ### CREATION de la variable havent_family (true/false) dÃ©terminant si le passager a de la famille ascendant/descendant
    newframe["havent_family"]=newframe.Parch ==0

    ### MODIFICATION du format de PClass, havent_family et Survived pour que les modalitÃ©s soient reconnues comme quali et non quanti
    newframe["Pclass"] = pandas.Categorical(newframe["Pclass"],ordered=False) # definition de Pclass comme catÃ©gorie pour manipuler les donnÃ©e
    newframe["Pclass"].cat.rename_categories(["c1", "c2", "c3"]) # conversion des donnÃ©es numÃ©riques en alpha pour faire les indicatrices

    newframe["havent_family"] = pandas.Categorical(newframe["havent_family"],ordered=False) # definition de havent_family comme catÃ©gorie pour manipuler les donnÃ©e
    newframe["havent_family"].cat.rename_categories(["N", "Y"]) # conversion des donnÃ©es numÃ©riques en alpha pour faire les indicatrices

    if with_surv_column == 1 : # permet de gÃ©rer si on travaille en training ou en predict
        newframe["Survived"] = pandas.Categorical(newframe["Survived"],ordered=False) # definition de Survived comme catÃ©gorie pour manipuler les donnÃ©e
        newframe["Survived"].cat.rename_categories(["N", "Y"]) # conversion des donnÃ©es numÃ©riques en alpha pour faire les indicatrices
    
    ### DISCRETISATION les variables quantitatives "age" et "fare" en 3 classes de mÃªme taille
    newframe["AgeQ"]=pandas.qcut(frame.Age,4,labels=["Ag1","Ag2","Ag3", "Ag4"])
    newframe["FareQ"]=pandas.qcut(frame.Fare,5,labels=["Fa1","Fa2","Fa3", "Fa4", "Fa5"])
    newframe_temp = newframe.drop(["Age", "Fare"], axis=1) # supprimer les variables quanti
    
    ### SUPPRESSION des colonnes non Ã©tudiÃ©es
    frame_temp = newframe_temp.drop(["PassengerId", "Parch", "Name", "SibSp", "Ticket", "Cabin"], axis=1)

    #CREATION d'une TABLE d'indicatrices en utilisation get_dummies sur chacune des data de framefin
    if with_surv_column == 1 : # permet de gÃ©rer si on travaille en training ou en predict
        frame_temp2=pandas.get_dummies(frame_temp[["Survived","havent_family","Pclass","Sex", "Embarked", "AgeQ","FareQ"]])
        frame_traitee = frame_temp2.drop(["Survived_0", "Sex_female", "havent_family_False"],axis =1) #suppression des variables binaires. conservation de l'indicatrice "survivant", "homme", ne possede pas de "famille"
    else :
        frame_temp2=pandas.get_dummies(frame_temp[["havent_family","Pclass","Sex", "Embarked", "AgeQ","FareQ"]])
        frame_traitee = frame_temp2.drop(["Sex_female", "havent_family_False"],axis =1) #suppression des variables binaires. conservation de l'indicatrice "survivant", "homme", ne possede pas de "famille"   
    return frame_traitee


# In[56]:

def echantil(frame) :# Extraction des Ã©chantillons d'apprentissage (tables Train) et de test (table Test).

    # suppression de la variable explicative dans la table de train (B = base de travail)
    B=frame.drop(["Survived_1"],axis=1) 

    # Variable Ã  modÃ©liser / rÃ©ponse
    r=frame["Survived_1"] # crÃ©ation d'une table contenant la variable Ã  expliquer

    # crÃ©ation des Ã©chantillons de training et de test sur chaque table
    import sklearn
    from sklearn.cross_validation import train_test_split
    B_train,B_test,r_train,r_test=train_test_split(B,r,test_size=0.2,random_state=11) # 20% en test
    
    return B_train,B_test,r_train,r_test


# In[57]:

def modele (all_base, frame_train_base, frame_train_reponse,frame_test_base, frame_test_reponse): # crÃ©ation du modele

    from sklearn.linear_model import LogisticRegression
    logit = LogisticRegression()
    frame_logit=logit.fit(frame_train_base, frame_train_reponse)

    if all_base == 0:
        print "score_test=" 
        print frame_logit.score(frame_test_base, frame_test_reponse)

    print "coeff_test="
    print frame_logit.coef_

    return frame_logit 


# In[58]:

### CrÃ©ation du Modele avec phase train/test (avec score)

frame_traitee = traitement(frame,1) # Preparation de la table - traitement des donnÃ©es / 1= with_surv_column
frame_train_base,frame_test_base,frame_train_reponse,frame_test_reponse = echantil(frame_traitee) # crÃ©ation des Ã©chantillons d'apprentissage (Base Train /rÃ©ponse train) et de test (base Test / rÃ©ponse test).
modele = modele (0, frame_train_base, frame_train_reponse,frame_test_base, frame_test_reponse) # crÃ©ation du modele


# In[52]:
### ERROR - CrÃ©ation du Modele avec table complÃ¨te (sans score)
#frame_traitee = traitement(frame,1) # Preparation de la table - traitement des donnÃ©es / 1= with_surv_column
#frame_train_base = frame_traitee.drop(["Survived_1"],axis=1)
#
#modele = modele (1, frame_train_base, frame_traitee["Survived_1"], 0, 0) # crÃ©ation du modele


# In[63]:

### Application du modele Ã  la table de competition (predict) & gÃ©nÃ©ration du rÃ©sultat
frame_traitee = traitement(titanic,0) # Preparation de la table - traitement des donnÃ©es / 1= with_surv_column

predictions = modele.predict(frame_traitee)

#ERROR with AZUREML print prediction result
import csv as csv

predictions_file = open("my_submission.csv", "wb")
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["PassengerId","Survived"])
open_file_object.writerows(zip(titanic['PassengerId'].values, predictions.astype(int)))
predictions_file.close()

ws





# In[ ]:



