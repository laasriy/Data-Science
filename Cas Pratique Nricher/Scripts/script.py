#Importation des libreries necessaires

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
import math
import os
from datetime import datetime
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import re
from unidecode import unidecode 
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import hstack
from sklearn.naive_bayes import MultinomialNB




# Importation de base de données
d = pd.read_excel('Data/20210614 Ecommerce sales.xlsb')


# Elimination des données manquantes
d.dropna(inplace=True)

# Caclul de prix avant l'application de prix de transport
d['Montant cmd sans transport'] = d['Montant cmd'] - d['Prix transport']


# Transformation du colonne de date
d['Date de commande'] = pd.to_datetime(d['Date de commande'], errors='coerce', unit='D', origin='1900-01-01')
d = d[~d['Date de commande'].isna()]
d['Date de commande'] = d['Date de commande'].dt.strftime('%Y-%m-%d')
d['Date de commande'] = d['Date de commande'].apply(lambda x: datetime.fromisoformat(x))

# Visualisation de données:

# Visualisation d'évolution des ventes dans le marketplace:
date = d.groupby(d["Date de commande"])['Montant cmd'].mean().reset_index()
dt = d.groupby(d["Date de commande"])['Montant cmd'].sum().reset_index()
date_vent = pd.DataFrame(date)
date_v = pd.DataFrame(dt)
date_v.rename(columns={"Montant cmd": "Montant cmd tot"}, inplace=True)

# Creation de sous graphe avec deux lignes et une colonne
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03)
# Ajout de premier y
fig.add_trace(go.Scatter(x=date_vent['Date de commande'], y=date_vent['Montant cmd'], mode='lines', name='Moyenne de CA'), row=1, col=1)
# Ajout du deuxième y
fig.add_trace(go.Scatter(x=date_v['Date de commande'], y=date_v['Montant cmd tot'], mode='lines', name='Total des ventes'), row=2, col=1)
# Ajout de légendes et titre
fig.update_layout(title='Évolution du CA')
fig.update_xaxes(title_text='Date de commande', row=2, col=1)
fig.update_yaxes(title_text='CA moyen', row=1, col=1)
fig.update_yaxes(title_text='CA total', row=2, col=1, secondary_y = False)
# Enregistrement de graphe
fig.write_html('Public/tendance_marketplace.html')

#-------------------------------------------------------------------

# Visualisation de ventes selon la nature:
nat = d.groupby(d['Nature'])['Montant cmd'].sum().reset_index()
nature = pd.DataFrame(nat)

fig = px.pie(nature, names = 'Nature', values = 'Montant cmd', color_discrete_sequence=px.colors.sequential.RdBu)
fig.update_traces(textposition='inside', textinfo='percent+label')
fig.update_layout(title='Ventes par Nature')
# Enregistrement de graphe
fig.write_html('Public/vente_nature.html')

#--------------------------------------------------------------------

# Visualisation des ventes par univers:
un = d.groupby(d['Univers'])['Montant cmd'].sum().reset_index()
univ = pd.DataFrame(un)
fig = px.pie(univ, names = 'Univers', values = 'Montant cmd')
fig.update_traces(textposition='inside', textinfo='percent+label')
fig.update_layout(title='Ventes par Univers')
# Enregistrement de graphe
fig.write_html('Public/vente_univers.html')

#------------------------------------------------------------------------
# Visualisation de Chiffre d'affaire par vendeur
# Groupement des données par vendeurs
vente_vendeurs = {}
for vendeur, vente, transport in zip(d['Vendeur'], d['Montant cmd sans transport'], d['Prix transport']):
    if vendeur in vente_vendeurs:
        vente_vendeurs[vendeur][0] += vente
        vente_vendeurs[vendeur][1] += transport
    else:
        vente_vendeurs[vendeur] = [vente, transport]

# Extraction des noms de vendeurs et le total des ventes
nom_vendeurs = list(vente_vendeurs.keys())
tot_ventes = [sum(values) for values in vente_vendeurs.values()]
tot_transports = [values[1] for values in vente_vendeurs.values()]

# Visualisation
plt.figure(figsize=(10, 6))
plt.bar(nom_vendeurs, tot_ventes, label='Ventes', color='blue')
plt.bar(nom_vendeurs, tot_transports, label='Transport', color='coral', alpha=0.7)
plt.xlabel('Vendeurs')
plt.ylabel('Montant')
plt.title('Composition du chiffre d\'affaires par vendeur')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
# Sauvgarde du graphe
plt.savefig('Public/composition_CA_par_vendeur.pdf')
#----------------------------------------------------------------------------
#Visualisation des ventes par vendeurs
v = d.groupby(d['Vendeur'])['Montant cmd'].sum().reset_index()
ven = pd.DataFrame(v)
fig = px.pie(ven, names = 'Vendeur', values = 'Montant cmd')
fig.update_traces(textposition='inside', textinfo='percent+label')
# Enregistrement de graphe
fig.write_html('Public/ventes_des_vendeur.html')

#----------------------------------------------------------------------------
# Visualisation de type des produits les plus vendus (Univers):
text = ' '.join(d['Univers'].dropna().drop_duplicates())

# Generate the word cloud
wordcloud = WordCloud(max_font_size=50, max_words=100,background_color="white").generate(text)

# Display the generated word cloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
# Sauvgarde du graphe
plt.savefig('Public/vendus_mots.pdf')

#==============================================================================================

# Algorithme d'identification des lignes mal classées:
"""
J'ai essayé de comparer la colonne de  'Nature' avec la colonne 'Libellé produit' pour identifier si la chaine de caractère qui se trouve dans le premièr colonne se trouve dans le deuxième.
Bien sur, j'ai tenu en compte que du faite que par forcement on va trouver la meme chose et que ce algo a ses limites comme le fait qu'il y a des synonymes des produits qui se trouve dans la colonne 'Libellé produit' et ne trouve pas dans la colonne 'Nature'

La fonction : existe prend comme input une colonne
dans la fonction on essaye de comparer les deux colonne.
L'OUTPUT: est une colonne 'String_Exists' où True si la ligne est bien classé ou False si la ligne n'est pas bien classée
"""
# Function to check if any part of 'Nature' exists in 'Libellé produit'
def existe(row):
    # Remove special characters and convert 'Nature' to lowercase
    nature_tokens = unidecode(str(row['Nature'])).split()

    # Remove special characters and convert 'Libellé produit' to lowercase
    phrase = unidecode(str(row['Libellé produit'])).lower()

    for token in nature_tokens:
        if token.lower() in phrase:
            return True

    return False
d['String_Exists'] = d.apply(existe, axis=1)

#Correction des lognes mal classée:
"""
Ce code effectue la préparation des données en convertissant les colonnes en types appropriés,
puis il encode les données de l'univers en utilisant LabelEncoder. Ensuite, il utilise la
vectorisation TF-IDF pour convertir les descriptions des produits en vecteurs numériques.
Les caractéristiques textuelles et encodées sont combinées en une seule matrice. Ensuite, il
divise les données en ensembles d'entraînement et de test, initialise un modèle Multinomial
Naive Bayes, et l'entraîne. Les prédictions sont faites sur l'ensemble de test, et seules
les catégories uniques sont conservées. Finalement, il effectue des prédictions sur l'ensemble
complet de données et ajoute les résultats dans une nouvelle colonne "Nature2" du DataFrame d'origine.
"""
#¨Préparation des variables
d['Libellé produit'] = d['Libellé produit'].astype(str)
X_text = d['Libellé produit'] 
X_univers = d['Univers']
y = d['Nature']

# Spécification des catégories qui existe dans la colonne 'Nature' et qu'on doit les conserver
unique_categories = y.unique()
# Coding des catégories de colonne 'Univers'
le = LabelEncoder()
X_univers_encoded = le.fit_transform(X_univers)

stopwords = stopwords.words('french')
# Vectorisation et traitement des variables
tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words = stopwords)
X_text_tfidf = tfidf_vectorizer.fit_transform(X_text)
X_univers_encoded = X_univers_encoded.reshape(-1, 1)  # Reshape the array
X_combined = hstack([X_text_tfidf, X_univers_encoded])

# Diviser les données en des sous échantillons
X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)

# Entrainement de modèle de classification Multinomial Naive Bayesien
classifier = MultinomialNB()
classifier.fit(X_train, y_train)

# Faire des prédictions qui vont servir à corriger notre colonne 'Nature' toujours en spésifiant qu'on ne peut pas inventer d'autres sous catégories en les garadant just les catégories existantes
y_pred = classifier.predict(X_test)
y_pred_filtered = [pred if pred in unique_categories else None for pred in y_pred]

y_pred = classifier.predict(X_combined)
# OUTPUT est dans une colonne pour regarder la différence entre les deux colonnes
d['Nature2'] = y_pred


#==============================================================================================
#Algorithme de traitement des matelas:
"""
Cette algorithme prend en compte que les matelas et fait ressourtir la dimension et les couleurs si il y en a dans la description de produit
La limite de ce algorithme est l'existence de plusieur couleurs que je n'avais pas pris en considération et du coup il y a des ligne où la couleur existe mais dans l'iuput il y en a pas.
"""
# prenant une règle pour chercher pour les matelas
pattern = re.compile(r'Matelas', re.IGNORECASE)

"""
Cette fonction prend comme input une colonne qui est 'Libellé produit' et cherche pour les matelas
L'OUTPUT est une colonne où True signifie que la ligne sert à une matela ou non (False)
"""
def matelas(row):
    phrase = str(row['Libellé produit'])
    return bool(pattern.search(phrase))
d['matelas'] = d.apply(matelas, axis=1)

mat = d[d['matelas'] == True ]
# Le traitement des colonnes:
dimension = r'(\d+(?:\.\d+)?(?:\s*[xX*]\s*\d+(?:\.\d+)?)+)\s*(cm|mm|inch|in)?'
couleurs = r'(blanc|noir|rouge|vert|bleu|jaune|rose|violet|marron|orange|gris)'

dimensions = []
couleur = []

"""
Après la spécification des règles de recherches dans la description de produits
j'ai initier une fonction qui sépare la chaine de caractère et qui détecte les dimension selon la règle qu'on a posé dans la liste dimension et couleur
L'OUTPUT sont deux colonnes de dimensions et couleurs
"""
for description in mat['Libellé produit']:
    dimension_matches = re.findall(dimension, description)
    couleur_matches = re.findall(couleurs, description, flags=re.IGNORECASE)

    if dimension_matches:
        dimensions.append(dimension_matches[0][0])
    else:
        dimensions.append(None)

    if couleur_matches:
        couleur.append(couleur_matches[0])
    else:
        couleur.append(None)

mat['Dimension'] = dimensions
mat['Couleur'] = couleur

# Division de la colonne de dimension en longeur et largeur si besoin:

mat['Longueur'] = None
mat['Largeur'] = None

for i, row in mat.iterrows():
    if row['Dimension'] is not None:
        dimensions = row['Dimension'].split('x')
        if len(dimensions) == 2:
            mat.at[i, 'Longueur'] = dimensions[0]
            mat.at[i, 'Largeur'] = dimensions[1]

#-----------------------------------------------------------------------------------------------------------------
# Visualisation des ventes et prix spécifiques des matelas
"""
Visualisation d'évolution de prix des matelas et prix de transport des matelas
"""
ma = mat.groupby('Date de commande')[['Montant cmd', 'Prix transport', 'Montant cmd sans transport']].sum().reset_index()

plt.figure(figsize=(12, 6))  

ax1 = plt.subplot()

ax1.plot(ma['Date de commande'], ma['Montant cmd'], color='b', label='Chiffre d\'affaires')
ax1.set_xlabel('Date de la commande')
ax1.set_ylabel('Chiffre d\'affaires', color='b')

ax2 = ax1.twinx()
ax2.plot(ma['Date de commande'], ma['Prix transport'], color='r', label='Frais de transport')
ax2.set_ylabel('Frais de transport', color='r')

lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines + lines2, labels + labels2, loc='upper left')

plt.title('Evolution des ventes des matelas et leur prix de transport')
plt.tight_layout() 
plt.savefig('Public/Evolution_prix_matelas.pdf')

#-------------------------------------------------------------
#Visualisation des ventes des matelas par vendeur:
vendeurs_matelas = mat.groupby('Vendeur')[['Montant cmd sans transport', 'Prix transport', 'Montant cmd']].sum().reset_index()

# Visualisation
plt.figure(figsize=(10, 6))
plt.bar(vendeurs_matelas['Vendeur'], vendeurs_matelas['Montant cmd sans transport'], label='Ventes', color='blue')
plt.bar(vendeurs_matelas['Vendeur'], vendeurs_matelas['Prix transport'], label='Transport', color='coral', alpha=0.7)
plt.xlabel('Vendeurs')
plt.ylabel('Montant')
plt.title("Composition du chiffre d'affaires par vendeur")
plt.legend()
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig('Public/matelas_vendeurs.pdf')

#-------------------------------------------------------------
# Influence de délai de transport sur les ventes des matelas

dure = mat.groupby('Délai transport annoncé')[['Montant cmd sans transport', 'Prix transport', 'Montant cmd']].sum().reset_index()

plt.figure(figsize=(10, 6))
ax = plt.subplot()
ax.plot(dure['Délai transport annoncé'], dure['Montant cmd'])

plt.xlabel('Durée de transport')
plt.ylabel('Ventes')
plt.title('Evolution des ventes en fonction de durée de transport')

plt.tight_layout()
plt.savefig('Public/delai_ventes.pdf')


#=============================================================================================================
d.to_excel('Public/DB.xlsx')
mat.to_excel('Public/matelas.xlsx')