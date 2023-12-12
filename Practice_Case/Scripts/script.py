#Importation of necessary libraries

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




# Importing the dataset
d = pd.read_excel(r'C:\Users\etulyon1\Desktop\Data-Science\Practice_Case\Data\20210614 Ecommerce sales.xlsb')


# Elimination of missing data 
d.dropna(inplace=True)

# Price calculation before the application of transport prices.
d['Montant cmd sans transport'] = d['Montant cmd'] - d['Prix transport']


# Transformation of the Date column
d['Date de commande'] = pd.to_datetime(d['Date de commande'], errors='coerce', unit='D', origin='1900-01-01')
d = d[~d['Date de commande'].isna()]
d['Date de commande'] = d['Date de commande'].dt.strftime('%Y-%m-%d')
d['Date de commande'] = d['Date de commande'].apply(lambda x: datetime.fromisoformat(x))

# Data Visualization

# Visualization of Sales Evolution in the Marketplace
date = d.groupby(d["Date de commande"])['Montant cmd'].mean().reset_index()
dt = d.groupby(d["Date de commande"])['Montant cmd'].sum().reset_index()
date_vent = pd.DataFrame(date)
date_v = pd.DataFrame(dt)
date_v.rename(columns={"Montant cmd": "Montant cmd tot"}, inplace=True)


fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03)

fig.add_trace(go.Scatter(x=date_vent['Date de commande'], y=date_vent['Montant cmd'], mode='lines', name='Moyenne de CA'), row=1, col=1)

fig.add_trace(go.Scatter(x=date_v['Date de commande'], y=date_v['Montant cmd tot'], mode='lines', name='Total des ventes'), row=2, col=1)

fig.update_layout(title='Évolution du CA')
fig.update_xaxes(title_text='Date de commande', row=2, col=1)
fig.update_yaxes(title_text='CA moyen', row=1, col=1)
fig.update_yaxes(title_text='CA total', row=2, col=1, secondary_y = False)

fig.write_html('Public/tendance_marketplace.html')

#-------------------------------------------------------------------

# Visualization of sales based on Nature (sub-category)
nat = d.groupby(d['Nature'])['Montant cmd'].sum().reset_index()
nature = pd.DataFrame(nat)

fig = px.pie(nature, names = 'Nature', values = 'Montant cmd', color_discrete_sequence=px.colors.sequential.RdBu)
fig.update_traces(textposition='inside', textinfo='percent+label')
fig.update_layout(title='Ventes par Nature')

fig.write_html('Public/vente_nature.html')

#--------------------------------------------------------------------

# Visualization of sales based on Univers (Category)
un = d.groupby(d['Univers'])['Montant cmd'].sum().reset_index()
univ = pd.DataFrame(un)
fig = px.pie(univ, names = 'Univers', values = 'Montant cmd')
fig.update_traces(textposition='inside', textinfo='percent+label')
fig.update_layout(title='Ventes par Univers')

fig.write_html('Public/vente_univers.html')

#------------------------------------------------------------------------
# Visualization of Sales for each seller

vente_vendeurs = {}
for vendeur, vente, transport in zip(d['Vendeur'], d['Montant cmd sans transport'], d['Prix transport']):
    if vendeur in vente_vendeurs:
        vente_vendeurs[vendeur][0] += vente
        vente_vendeurs[vendeur][1] += transport
    else:
        vente_vendeurs[vendeur] = [vente, transport]


nom_vendeurs = list(vente_vendeurs.keys())
tot_ventes = [sum(values) for values in vente_vendeurs.values()]
tot_transports = [values[1] for values in vente_vendeurs.values()]


plt.figure(figsize=(10, 6))
plt.bar(nom_vendeurs, tot_ventes, label='Ventes', color='blue')
plt.bar(nom_vendeurs, tot_transports, label='Transport', color='coral', alpha=0.7)
plt.xlabel('Vendeurs')
plt.ylabel('Montant')
plt.title('Composition du chiffre d\'affaires par vendeur')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig('Public/composition_CA_par_vendeur.pdf')
#----------------------------------------------------------------------------
# Sales Visualization by Sellers
v = d.groupby(d['Vendeur'])['Montant cmd'].sum().reset_index()
ven = pd.DataFrame(v)
fig = px.pie(ven, names = 'Vendeur', values = 'Montant cmd')
fig.update_traces(textposition='inside', textinfo='percent+label')

fig.write_html('Public/ventes_des_vendeur.html')

#----------------------------------------------------------------------------
# Visualization of the most sold products by Univers (Category)
text = ' '.join(d['Univers'].dropna().drop_duplicates())


wordcloud = WordCloud(max_font_size=50, max_words=100,background_color="white").generate(text)


plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")

plt.savefig('Public/vendus_mots.pdf')

#==============================================================================================

# Algorithme to indentify the misclassified lines or products
"""
I tried to compare the 'Nature' column with the 'Product Description' column to identify whether the string in the first column is present in the second one.

Of course, I took into account that we might not always find an exact match, and this algorithm has its limitations, such as the existence of product synonyms in the 'Product Description' column that may not be present in the 'Nature' column.

The function "exists" takes a column as input, and within the function, we attempt to compare the two columns. The OUTPUT is a 'String_Exists' column where it is True if the row is properly classified and False if the row is not properly classified.
"""

def existe(row):
    
    nature_tokens = unidecode(str(row['Nature'])).split()

    
    phrase = unidecode(str(row['Libellé produit'])).lower()

    for token in nature_tokens:
        if token.lower() in phrase:
            return True

    return False
d['String_Exists'] = d.apply(existe, axis=1)

# Correction of misclassified line or products
"""
This code performs data preprocessing by converting columns into appropriate types, then encodes the universe data using LabelEncoder. 
Next, it utilizes TF-IDF vectorization to convert product descriptions into numerical vectors. 
Textual and encoded features are combined into a single matrix. 
It then splits the data into training and testing sets, initializes a Multinomial Naive Bayes model, and trains it. 
Predictions are made on the test set, retaining only unique categories. 
Finally, it makes predictions on the entire dataset and adds the results to a new "Nature2" column in the original DataFrame.
"""

d['Libellé produit'] = d['Libellé produit'].astype(str)
X_text = d['Libellé produit'] 
X_univers = d['Univers']
y = d['Nature']


unique_categories = y.unique()

le = LabelEncoder()
X_univers_encoded = le.fit_transform(X_univers)

stopwords = stopwords.words('french')

tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words = stopwords)
X_text_tfidf = tfidf_vectorizer.fit_transform(X_text)
X_univers_encoded = X_univers_encoded.reshape(-1, 1) 
X_combined = hstack([X_text_tfidf, X_univers_encoded])


X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)


classifier = MultinomialNB()
classifier.fit(X_train, y_train)


y_pred = classifier.predict(X_test)
y_pred_filtered = [pred if pred in unique_categories else None for pred in y_pred]

y_pred = classifier.predict(X_combined)

d['Nature2'] = y_pred


#==============================================================================================
# Mattress processing algorithm:
"""
This algorithm takes into account the mattresses and highlights the dimensions and colors if there are any in the product description. 
The limitation of this algorithm is the existence of multiple colors that I did not take into consideration, and as a result, there are lines where the color exists, but it is not present in the input.
"""

pattern = re.compile(r'Matelas', re.IGNORECASE)

"""
This function takes as input a column that is the 'Product Label' and searches for mattresses. 
The OUTPUT is a column where True means that the row corresponds to a mattress, and False means it does not.
"""
def matelas(row):
    phrase = str(row['Libellé produit'])
    return bool(pattern.search(phrase))
d['matelas'] = d.apply(matelas, axis=1)

mat = d[d['matelas'] == True ]

dimension = r'(\d+(?:\.\d+)?(?:\s*[xX*]\s*\d+(?:\.\d+)?)+)?(?:\s*[xX*]\s*\d+(?:\.\d+)?)+)\s*(cm|mm|inch|in)?'
couleurs = r'(blanc|noir|rouge|vert|bleu|jaune|rose|violet|marron|orange|gris)'

dimensions = []
couleur = []

"""
After specifying the search rules in the product description, I initiated a function that separates the character string and detects the dimensions according to the rules set in the dimension and color list. 
The OUTPUT consists of two columns of dimensions and colors.
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

# 

mat['Longueur'] = None
mat['Largeur'] = None
mat['Epaisseur']

for i, row in mat.iterrows():
    if row['Dimension'] is not None:
        dimensions = row['Dimension'].split('x')
        if len(dimensions) == 3:
            mat.at[i, 'Longueur'] = dimensions[0]
            mat.at[i, 'Largeur'] = dimensions[1]
            mat.at[i, 'Largeur'] = dimensions[2]

#-----------------------------------------------------------------------------------------------------------------
# Visualization of mattress sales and specific prices
"""
Visualization of mattress price trends and mattress transportation costs.
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
# Visualization of mattress sales by salesperson:
vendeurs_matelas = mat.groupby('Vendeur')[['Montant cmd sans transport', 'Prix transport', 'Montant cmd']].sum().reset_index()


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
# Influence of Transport Time on Mattress Sales

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