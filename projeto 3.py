import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.figure_factory as ff
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import io
import base64

# Carregar os dados
df = pd.read_csv('ecommerce_estatistica (1).csv')
print(df.columns)
# Preparar os dados
X = df.drop('Qtd_Vendidos', axis=1)
y = df['Qtd_Vendidos']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Modelos
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)
y_pred_log = log_model.predict(X_test)

tree_model = DecisionTreeClassifier()
tree_model.fit(X_train, y_train)
y_pred_tree = tree_model.predict(X_test)

lin_model = LinearRegression()
lin_model.fit(X_train, y_train)
y_pred_lin = lin_model.predict(X_test)


# Funções auxiliares para gerar imagens dos gráficos
def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax)
    ax.set_title(title)
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode()
    return f"data:image/png;base64,{encoded}"


# Gráficos
conf_matrix_log = plot_confusion_matrix(y_test, y_pred_log, "Matriz de Confusão - Regressão Logística")
conf_matrix_tree = plot_confusion_matrix(y_test, y_pred_tree, "Matriz de Confusão - Árvore de Decisão")

# Heatmaps
corr_pearson = df.corr(method='pearson')
corr_spearman = df.corr(method='spearman')

heatmap_pearson = px.imshow(corr_pearson, text_auto=True, title="Correlação Pearson")
heatmap_spearman = px.imshow(corr_spearman, text_auto=True, title="Correlação Spearman")

# Aplicação Dash
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Dashboard de Visualização Estatística"),

    html.H2("Matriz de Confusão - Regressão Logística"),
    html.Img(src=conf_matrix_log),

    html.H2("Matriz de Confusão - Árvore de Decisão"),
    html.Img(src=conf_matrix_tree),

    html.H2("Gráfico de Regressão Linear"),
    dcc.Graph(
        figure=px.scatter(x=y_test, y=y_pred_lin, labels={'x': 'Valor Real', 'y': 'Valor Predito'},
                          title="Regressão Linear")
    ),

    html.H2("Heatmap de Correlação Pearson"),
    dcc.Graph(figure=heatmap_pearson),

    html.H2("Heatmap de Correlação Spearman"),
    dcc.Graph(figure=heatmap_spearman),

    html.H2("Visualização Interativa de Correlação Pearson"),
    dcc.Dropdown(
        id='feature-x',
        options=[{'label': col, 'value': col} for col in df.columns],
        value=df.columns[0]
    ),
    dcc.Dropdown(
        id='feature-y',
        options=[{'label': col, 'value': col} for col in df.columns],
        value=df.columns[1]
    ),
    dcc.Graph(id='scatter-pearson'),

    html.H2("Visualização Interativa de Correlação Spearman"),
    dcc.Graph(id='scatter-spearman')
])


@app.callback(
    Output('scatter-pearson', 'figure'),
    [Input('feature-x', 'value'),
     Input('feature-y', 'value')]
)
def update_scatter_pearson(feature_x, feature_y):
    fig = px.scatter(df, x=feature_x, y=feature_y, trendline="ols",
                     title=f"Correlação Pearson entre {feature_x} e {feature_y}")
    return fig


@app.callback(
    Output('scatter-spearman', 'figure'),
    [Input('feature-x', 'value'),
     Input('feature-y', 'value')]
)
def update_scatter_spearman(feature_x, feature_y):
    df_ranked = df.rank()
    fig = px.scatter(df_ranked, x=feature_x, y=feature_y, trendline="ols",
                     title=f"Correlação Spearman entre {feature_x} e {feature_y}")
    return fig


# Executar o app
if __name__ == '__main__':
    app.run_server(debug=True)
