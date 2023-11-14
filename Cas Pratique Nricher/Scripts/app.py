import dash
from dash import dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import pandas as pd

# Charger votre jeu de données
df = pd.read_excel('Data/20210614 Ecommerce sales.xlsb')

# Créer une application Dash
app = dash.Dash(__name__)

# Styles CSS pour améliorer la présentation
app.css.append_css({'external_url': 'https://codepen.io/chriddyp/pen/bWLwgP.css'})

# Mise en page de l'application
app.layout = html.Div([
    html.H1("Tableau de bord d'analyse des ventes en ligne"),
    dcc.Graph(id='tendance-marketplace'),
    dcc.Graph(id='vente-nature'),
    dcc.Graph(id='vente-univers'),
    dcc.Graph(id='ventes-par-vendeur'),
    dcc.Graph(id='ventes-des-vendeur'),
    html.Img(id='word-cloud'),
    dcc.Graph(id='evolution-prix-matelas'),
    dcc.Graph(id='matelas-vendeurs'),
    dcc.Graph(id='delai-ventes')
])

# Callbacks pour afficher les graphiques
@app.callback(
    Output('tendance-marketplace', 'figure'),
    Output('vente-nature', 'figure'),
    Output('vente-univers', 'figure'),
    Output('ventes-par-vendeur', 'figure'),
    Output('ventes-des-vendeur', 'figure'),
    Output('word-cloud', 'src'),
    Output('evolution-prix-matelas', 'figure'),
    Output('matelas-vendeurs', 'figure'),
    Output('delai-ventes', 'figure'),
    Input('tendance-marketplace', 'value'))
def update_graph(selected_value):
    # Placez ici le code pour mettre à jour les graphiques en fonction de la valeur sélectionnée

    if __name__ == '__main__':
        app.run_server(debug=True)
