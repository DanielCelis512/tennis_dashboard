import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.metrics import r2_score, mean_squared_error, confusion_matrix, accuracy_score

# ── DATOS ────────────────────────────────────────────────
np.random.seed(42)
primer_servicio = np.random.normal(60, 10, 100)
puntos_ganados  = primer_servicio * 1.5 + np.random.normal(0, 10, 100)

df = pd.DataFrame({
    "Primer_Servicio_%": primer_servicio,
    "Puntos_Ganados":    puntos_ganados
})
df["Aces"]               = np.random.normal(5,   2,  100)
df["Dobles_Faltas"]      = np.random.normal(3,   1,  100)
df["Segundo_Servicio_%"] = np.random.normal(50,  8,  100)
df["Velocidad_Servicio"] = np.random.normal(180, 20, 100)
df["Errores_No_Forzados"]    = np.random.normal(15,  5,  100)
df["Break_Points_Salvados"]  = np.random.normal(60,  15, 100)

umbral = df["Puntos_Ganados"].mean()
df["Resultado"] = (df["Puntos_Ganados"] > umbral).astype(int)

# ── MODELOS ──────────────────────────────────────────────
X3 = df[["Primer_Servicio_%", "Aces", "Segundo_Servicio_%"]]
X6 = df[["Primer_Servicio_%", "Aces", "Segundo_Servicio_%",
          "Velocidad_Servicio", "Errores_No_Forzados", "Break_Points_Salvados"]]
y  = df["Puntos_Ganados"]

X_train3, X_test3, y_train3, y_test3 = train_test_split(X3, y, test_size=0.3, random_state=42)
X_train6, X_test6, y_train6, y_test6 = train_test_split(X6, y, test_size=0.3, random_state=42)

lin3 = LinearRegression().fit(X_train3, y_train3)
lasso = Lasso(alpha=1.0).fit(X_train6, y_train6)

y_pred3    = lin3.predict(X_test3)
y_pred_las = lasso.predict(X_test6)

X_log = df[["Primer_Servicio_%"]]
y_log = df["Resultado"]
Xl_train, Xl_test, yl_train, yl_test = train_test_split(X_log, y_log, test_size=0.3, random_state=42)
log_model = LogisticRegression(C=1.0, solver="lbfgs").fit(Xl_train, yl_train)
y_pred_log = log_model.predict(Xl_test)

# Métricas para comparación
modelos_nombres = ["LinearReg\n3 vars", "Ridge α=1", "Ridge α=10", "Lasso α=0.1", "Lasso α=1"]
configs = [
    LinearRegression(),
    Ridge(alpha=1.0), Ridge(alpha=10.0),
    Lasso(alpha=0.1), Lasso(alpha=1.0)
]
r2_vals, mse_vals = [], []
for m in configs:
    m.fit(X_train6, y_train6)
    yp = m.predict(X_test6)
    r2_vals.append(round(r2_score(y_test6, yp), 4))
    mse_vals.append(round(mean_squared_error(y_test6, yp), 4))

# ── APP ──────────────────────────────────────────────────
app = Dash(__name__)

COLORS = {
    "bg":       "#0f0f1a",
    "card":     "#1a1a2e",
    "violet":   "#7c3aed",
    "cyan":     "#06b6d4",
    "text":     "#e2e8f0",
    "subtext":  "#94a3b8",
}

def card(children):
    return html.Div(children, style={
        "backgroundColor": COLORS["card"],
        "borderRadius": "12px",
        "padding": "20px",
        "margin": "10px",
    })

app.layout = html.Div(style={"backgroundColor": COLORS["bg"], "minHeight": "100vh", "padding": "20px"}, children=[

    # TÍTULO
    html.Div([
        html.H1("🎾 Dashboard – Análisis de Tenis", style={"color": COLORS["violet"], "textAlign": "center"}),
        html.P("Daniel Celis · Programación para Ciencia de Datos II",
               style={"color": COLORS["subtext"], "textAlign": "center"}),
    ]),

    # MÉTRICAS CLAVE
    html.Div(style={"display": "flex", "justifyContent": "center", "flexWrap": "wrap"}, children=[
        card([html.H3("r = 0.7936", style={"color": COLORS["cyan"]}),
              html.P("Correlación de Pearson", style={"color": COLORS["subtext"]})]),
        card([html.H3("R² = 0.574", style={"color": COLORS["cyan"]}),
              html.P("Regresión Lineal (3 vars)", style={"color": COLORS["subtext"]})]),
        card([html.H3("R² = 0.5498", style={"color": COLORS["cyan"]}),
              html.P("Lasso α=1.0 (6 vars)", style={"color": COLORS["subtext"]})]),
        card([html.H3("73.3%", style={"color": COLORS["cyan"]}),
              html.P("Precisión Logística", style={"color": COLORS["subtext"]})]),
    ]),

    # GRÁFICA 1: SCATTER INTERACTIVO
    card([
        html.H3("Relación: Primer Servicio vs Puntos Ganados", style={"color": COLORS["text"]}),
        html.P("Filtra por rango de primer servicio:", style={"color": COLORS["subtext"]}),
        dcc.RangeSlider(
            id="slider-servicio",
            min=int(df["Primer_Servicio_%"].min()),
            max=int(df["Primer_Servicio_%"].max()),
            step=1,
            value=[int(df["Primer_Servicio_%"].min()), int(df["Primer_Servicio_%"].max())],
            marks={i: {"label": str(i), "style": {"color": COLORS["subtext"]}}
                   for i in range(30, 85, 10)},
        ),
        dcc.Graph(id="scatter-plot"),
    ]),

    # GRÁFICA 2: REAL VS PREDICHO
    card([
        html.H3("Regresión Lineal – Real vs Predicho", style={"color": COLORS["text"]}),
        html.P("Selecciona el modelo:", style={"color": COLORS["subtext"]}),
        dcc.Dropdown(
            id="dropdown-modelo",
            options=[
                {"label": "LinearRegression (3 variables)", "value": "linear"},
                {"label": "Lasso α=1.0 (6 variables)",      "value": "lasso"},
            ],
            value="linear",
            style={"backgroundColor": COLORS["card"], "color": "#000"}
        ),
        dcc.Graph(id="real-vs-pred"),
    ]),

    # GRÁFICA 3: COMPARACIÓN DE MODELOS
    card([
        html.H3("Comparación de Modelos – Ridge y Lasso", style={"color": COLORS["text"]}),
        html.P("Selecciona la métrica a visualizar:", style={"color": COLORS["subtext"]}),
        dcc.RadioItems(
            id="radio-metrica",
            options=[{"label": " R²", "value": "r2"}, {"label": " MSE", "value": "mse"}],
            value="r2",
            inline=True,
            style={"color": COLORS["text"], "marginBottom": "10px"}
        ),
        dcc.Graph(id="comparacion-modelos"),
    ]),

    # GRÁFICA 4: MATRIZ DE CONFUSIÓN
    card([
        html.H3("Regresión Logística – Matriz de Confusión", style={"color": COLORS["text"]}),
        dcc.Graph(id="confusion-matrix"),
    ]),

])

# ── CALLBACKS ────────────────────────────────────────────

@app.callback(Output("scatter-plot", "figure"), Input("slider-servicio", "value"))
def update_scatter(rango):
    filtered = df[(df["Primer_Servicio_%"] >= rango[0]) & (df["Primer_Servicio_%"] <= rango[1])]
    fig = px.scatter(filtered, x="Primer_Servicio_%", y="Puntos_Ganados",
                     trendline="ols", color_discrete_sequence=[COLORS["violet"]],
                     labels={"Primer_Servicio_%": "Primer Servicio (%)", "Puntos_Ganados": "Puntos Ganados"})
    fig.update_layout(paper_bgcolor=COLORS["card"], plot_bgcolor=COLORS["card"],
                      font_color=COLORS["text"])
    fig.update_traces(marker=dict(size=8, opacity=0.8))
    return fig

@app.callback(Output("real-vs-pred", "figure"), Input("dropdown-modelo", "value"))
def update_real_pred(modelo):
    if modelo == "linear":
        y_real, y_pred = y_test3, y_pred3
        r2 = round(r2_score(y_real, y_pred), 3)
        label = f"LinearRegression — R²={r2}"
    else:
        y_real, y_pred = y_test6, y_pred_las
        r2 = round(r2_score(y_real, y_pred), 3)
        label = f"Lasso α=1.0 — R²={r2}"

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_real, y=y_pred, mode="markers",
                             marker=dict(color=COLORS["violet"], size=9, opacity=0.8),
                             name="Predicciones"))
    fig.add_trace(go.Scatter(x=[y_real.min(), y_real.max()],
                             y=[y_real.min(), y_real.max()],
                             mode="lines", line=dict(color=COLORS["cyan"], dash="dash"),
                             name="Predicción perfecta"))
    fig.update_layout(title=label, xaxis_title="Valores Reales", yaxis_title="Valores Predichos",
                      paper_bgcolor=COLORS["card"], plot_bgcolor=COLORS["card"],
                      font_color=COLORS["text"])
    return fig

@app.callback(Output("comparacion-modelos", "figure"), Input("radio-metrica", "value"))
def update_comparacion(metrica):
    valores = r2_vals if metrica == "r2" else mse_vals
    titulo  = "Comparación R²" if metrica == "r2" else "Comparación MSE"
    fig = px.bar(x=modelos_nombres, y=valores, color=valores,
                 color_continuous_scale=["#4c1d95", "#7c3aed", "#06b6d4"],
                 labels={"x": "Modelo", "y": metrica.upper()})
    fig.update_layout(title=titulo, paper_bgcolor=COLORS["card"],
                      plot_bgcolor=COLORS["card"], font_color=COLORS["text"],
                      showlegend=False)
    return fig

@app.callback(Output("confusion-matrix", "figure"), Input("dropdown-modelo", "value"))
def update_confusion(_):
    cm = confusion_matrix(yl_test, y_pred_log)
    fig = px.imshow(cm, text_auto=True, color_continuous_scale="Purples",
                    x=["Predicho 0", "Predicho 1"],
                    y=["Real 0", "Real 1"],
                    labels=dict(color="Casos"))
    fig.update_layout(paper_bgcolor=COLORS["card"], font_color=COLORS["text"])
    return fig

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
    