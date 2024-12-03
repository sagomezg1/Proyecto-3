import pandas as pd
import plotly.express as px
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import torch
import joblib
import numpy as np

# Cargar los datos
df = pd.read_csv(r"C:\Users\Nicolas\Desktop\Proyecto 3 analitica comp\dfFinal.csv")

# Crear columnas consolidadas para nivel educativo
df['Nivel_Academico_Padre'] = df.filter(like='FAMI_EDUCACIONPADRE_').idxmax(axis=1).str.replace('FAMI_EDUCACIONPADRE_', '')
df['Nivel_Academico_Madre'] = df.filter(like='FAMI_EDUCACIONMADRE_').idxmax(axis=1).str.replace('FAMI_EDUCACIONMADRE_', '')

# Ruta al modelo y al scaler
MODEL_PATH = r"C:\Users\Nicolas\Desktop\Proyecto 3 analitica comp\Modelos\Global\Global.pth"
SCALER_PATH = r"C:\Users\Nicolas\Desktop\Proyecto 3 analitica comp\Modelos\Global\Global.pkl"

# Cargar el modelo
class RegressionModel(torch.nn.Module):
    def __init__(self, input_dim):
        super(RegressionModel, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, 64)
        self.fc2 = torch.nn.Linear(64, 64)
        self.fc3 = torch.nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Crear una instancia del modelo con input_dim=5
model = RegressionModel(input_dim=5)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

# Cargar el scaler
scaler = joblib.load(SCALER_PATH)

# Crear la aplicación Dash
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Analítica de resultados Saber 11"),

    # Gráfica 1: Total de puntajes vs Nivel Académico del Padre
    dcc.Graph(
        id='grafico1',
        figure=px.bar(
            df.groupby('Nivel_Academico_Padre')['PUNT_GLOBAL'].sum().reset_index(),
            x='Nivel_Academico_Padre',
            y='PUNT_GLOBAL',
            title='Total de Puntajes Saber 11 por Nivel Académico del Padre',
            labels={'PUNT_GLOBAL': 'Total de Puntaje', 'Nivel_Academico_Padre': 'Nivel Académico del Padre'},
            text_auto=True
        )
    ),

    # Gráfica 2: Total de puntajes vs Nivel Académico de la Madre
    dcc.Graph(
        id='grafico2',
        figure=px.bar(
            df.groupby('Nivel_Academico_Madre')['PUNT_GLOBAL'].sum().reset_index(),
            x='Nivel_Academico_Madre',
            y='PUNT_GLOBAL',
            title='Total de Puntajes Saber 11 por Nivel Académico de la Madre',
            labels={'PUNT_GLOBAL': 'Total de Puntaje', 'Nivel_Academico_Madre': 'Nivel Académico de la Madre'},
            text_auto=True
        )
    ),

    # Nueva Gráfica: Gráfica de caja (cuantiles) por nivel educativo de la madre
    dcc.Graph(
        id='grafico_cuantiles_madre',
        figure=px.box(
            df,
            x='Nivel_Academico_Madre',
            y='PUNT_GLOBAL',
            title='Distribución Cuantil del Puntaje Saber 11 por Nivel Educativo de la Madre',
            labels={'PUNT_GLOBAL': 'Puntaje Saber 11', 'Nivel_Academico_Madre': 'Nivel Académico de la Madre'},
            points='all'  # Muestra todos los puntos para mayor detalle
        )
    ),

    # Sección de predicción
    html.H1("Selecciona las Variables para la Predicción del Puntaje Saber 11"),

    # Selector 1: COLE_CARACTER_ACADÉMICO
    html.Div([
        html.Label("¿El colegio es académico?"),
        dcc.Dropdown(
            id='selector_cole_caracter_academico',
            options=[
                {'label': 'Sí (Académico)', 'value': 1},
                {'label': 'No (No Académico)', 'value': 0},
            ],
            value=0  # Valor por defecto
        ),
    ], style={'padding': '10px'}),

    # Selector 2: COLE_BILINGUE
    html.Div([
        html.Label("¿El colegio es bilingüe?"),
        dcc.Dropdown(
            id='selector_cole_bilingue',
            options=[
                {'label': 'Sí (Bilingüe)', 'value': 1},
                {'label': 'No (No Bilingüe)', 'value': 0},
            ],
            value=0  # Valor por defecto
        ),
    ], style={'padding': '10px'}),

    # Selector 3: COLE_JORNADA_COMPLETA
    html.Div([
        html.Label("¿El colegio tiene jornada completa?"),
        dcc.Dropdown(
            id='selector_cole_jornada_completa',
            options=[
                {'label': 'Sí (Jornada Completa)', 'value': 1},
                {'label': 'No (No Jornada Completa)', 'value': 0},
            ],
            value=0  # Valor por defecto
        ),
    ], style={'padding': '10px'}),

    # Selector 4: FAMI_TIENEAUTOMOVIL
    html.Div([
        html.Label("¿La familia tiene automóvil?"),
        dcc.Dropdown(
            id='selector_fami_tieneautomovil',
            options=[
                {'label': 'Sí (Tiene Automóvil)', 'value': 1},
                {'label': 'No (No Tiene Automóvil)', 'value': 0},
            ],
            value=0  # Valor por defecto
        ),
    ], style={'padding': '10px'}),

    # Selector 5: FAMI_TIENECOMPUTADOR
    html.Div([
        html.Label("¿La familia tiene computadora?"),
        dcc.Dropdown(
            id='selector_fami_tiencomputador',
            options=[
                {'label': 'Sí (Tiene Computadora)', 'value': 1},
                {'label': 'No (No Tiene Computadora)', 'value': 0},
            ],
            value=0  # Valor por defecto
        ),
    ], style={'padding': '10px'}),

    # Botón para hacer la predicción
    html.Div([
        html.Button('Calcular puntaje global Saber 11', id='boton_calcular', n_clicks=0),
        html.Div(id='resultado_prediccion', style={'padding': '20px'})
    ]),
])

@app.callback(
    Output('resultado_prediccion', 'children'),
    Input('boton_calcular', 'n_clicks'),
    Input('selector_cole_caracter_academico', 'value'),
    Input('selector_cole_bilingue', 'value'),
    Input('selector_cole_jornada_completa', 'value'),
    Input('selector_fami_tieneautomovil', 'value'),
    Input('selector_fami_tiencomputador', 'value')
)
def predecir_puntaje(n_clicks, cole_caracter_academico, cole_bilingue, cole_jornada_completa,
                     fami_tieneautomovil, fami_tiencomputador):
    if n_clicks == 0:
        return "Selecciona las variables y haz clic en 'Calcular puntaje global Saber 11'."

    # Crear un arreglo con las variables seleccionadas
    input_data = np.array([[cole_caracter_academico, cole_bilingue, cole_jornada_completa,
                            fami_tieneautomovil, fami_tiencomputador]])

    # Escalar las variables
    input_scaled = scaler.transform(input_data)

    # Convertir a tensor de PyTorch
    input_tensor = torch.tensor(input_scaled, dtype=torch.float32)

    # Hacer la predicción
    with torch.no_grad():
        prediccion = model(input_tensor).item()

    return f"El puntaje global estimado del Saber 11 es: {prediccion:.2f}"

if __name__ == '__main__':
    app.run_server(debug=True)
