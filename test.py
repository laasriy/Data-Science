from dash import Dash, dcc, html, Input, Output
import plotly.express as px
from base64 import b64encode
import numpy as np
np.random.seed(1)

app = Dash(__name__)


app.layout = html.Div([
    html.H4('Rendering options of plots in Dash '),
    html.P("Choose render option:"),
    dcc.RadioItems(
        id='render-option',
        options=['interactive', 'image'],
        value='image'
    ),
    html.Div(id='output'),
])


@app.callback(
    Output("output", "children"), 
    Input('render-option', 'value'))
def display_graph(render_option):
    x, y, sz, cl = np.random.rand(4, 100) # replace with your own data source
    fig = px.scatter(x=x, y=y, size=sz, color=cl)

    if render_option == 'image':
        img_bytes = fig.to_image(format="png")
        encoding = b64encode(img_bytes).decode()
        img_b64 = "data:image/png;base64," + encoding
        return html.Img(src=img_b64, style={'height': '500px'})
    else:
        return dcc.Graph(figure=fig)


app.run_server(debug=True)