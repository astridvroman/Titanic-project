######## IMPORT DE LA TABLE DE DONNEES ###############
from azureml import Workspace

ws = Workspace(
    workspace_id='5fea9ff155714fa59da6bc15e9cfddc0',
    authorization_token='e1ed5eab2ac040c4a1c8ed5ad14621fb',
    endpoint='https://studioapi.azureml.net'
)
ds = ws.datasets['train.csv']
frame = ds.to_dataframe()
import pandas

####### FONCTION QUI PERMET DE TESTER SI UNE VALEUR INDEXEE EN i EST UNE DONNEE MANQUANTE ##############
def select_isnull(i) : # i index de la série étudiée; cet index sera placé dans un des groupes
    if pandas.isnull(frame[x][i]) == True:
        return 'valeur Nan' # je définie le nom du groupe
    else:
        return 'valeur non Nan'
        
####### COMPTER LES DONNEES MANQUANTES DANS CHQUE COLONNE DE FRAME ##############
for x in frame.columns : # x itère sur chaque colonne de frame
    
    nan_group = frame[x].groupby(select_isnull)
        #nan group est un dictionnaire avec 2 groupes avec dedans les indexes correspondants
        #groupby permet de créer des groupes selon le critère établi par la fonction select_isnull
        #groupby scanne chaque index de la série visée et non directement la valeur
    
    if "valeur Nan" not in nan_group.groups : # certaine séries n'ont pas pas de valeur nulle
        print x + " nb valeur Nan = 0" + "/ " + str(len(frame[x]))
    else :
        print x + " nb valeur Nan = " + str(len(nan_group.groups["valeur Nan"]))+ "/ " + str(len(frame[x])) # la longeur du groupe me permet de compter le nb d'index à l"intérieur
