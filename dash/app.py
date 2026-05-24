"""
Tablero Saber 11 — Córdoba
Análisis predictivo · Secretaría de Educación del Departamento de Córdoba
"""

import os
import json
import numpy as np
import pandas as pd
import dash
from dash import dcc, html, Input, Output, State
from dash.exceptions import PreventUpdate

try:
    import joblib
except Exception:
    joblib = None

try:
    from tensorflow import keras
except Exception:
    keras = None
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ─────────────────────────────────────────────────────────────────────
# 1. PALETTE & PLOTLY DEFAULTS
# ─────────────────────────────────────────────────────────────────────
NAV   = '#1a2c42'
NAVY  = '#1e3a5f'
BLUE  = '#2e6da4'
MID   = '#5b9bd5'
LGRAY = '#9ca3af'
RED   = '#b03a2e'
GRN   = '#2d6a4f'
AMB   = '#92400e'
TEXT  = '#111827'
SUB   = '#6b7280'
BORD  = '#e5e7eb'

_L = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family='-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif',
              color=TEXT, size=12),
    margin=dict(l=10, r=14, t=44, b=10),
    legend=dict(bgcolor='rgba(0,0,0,0)', borderwidth=0, font=dict(size=11)),
)
_AX = dict(gridcolor='#efefef', linecolor='#efefef', tickfont=dict(size=11))

_CFG = dict(
    displayModeBar=True,
    modeBarButtonsToRemove=[
        'zoom2d', 'pan2d', 'select2d', 'lasso2d',
        'zoomIn2d', 'zoomOut2d', 'autoScale2d', 'toggleSpikelines',
    ],
    displaylogo=False,
    toImageButtonOptions=dict(format='png', filename='saber11_cordoba', scale=2),
)


# ─────────────────────────────────────────────────────────────────────
# 2. CSV LOADING & AGGREGATION
# ─────────────────────────────────────────────────────────────────────
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_CSV = os.path.join(_ROOT, 'Resultados_ICFES_Cordoba_clean.csv')

_XGB_MODEL_PATH = os.path.join(_ROOT, 'modelo_final_icfes_xgboost.joblib')
_NN_MODEL_PATH = os.path.join(_ROOT, 'modelo_final_icfes.keras')
_COLUMNAS_PATH = os.path.join(_ROOT, 'columnas_modelo.json')
_CATEGORIAS_PATH = os.path.join(_ROOT, 'categorias_modelo.json')
_THRESHOLD_XGB_PATH = os.path.join(_ROOT, 'threshold_xgboost.json')

# v2 models (puntaje global + inglés A-)
_DASH_DIR   = os.path.dirname(os.path.abspath(__file__))
_MODELS_DIR = os.path.join(_DASH_DIR, '..', 'Tarea4_Modelos', 'modelos')


def _safe_load_json(path, fallback):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f'[!] No se pudo cargar {os.path.basename(path)} — {e}')
        return fallback


COLUMNAS_MODELO = _safe_load_json(_COLUMNAS_PATH, [])

CATEGORIAS_MODELO = _safe_load_json(
    _CATEGORIAS_PATH,
    {
        'categorical_features': {},
        'numeric_features': {}
    }
)

THRESHOLD_XGB_INFO = _safe_load_json(
    _THRESHOLD_XGB_PATH,
    {'optimized_threshold': 0.3103}
)

MAT_XGB_THRESHOLD = float(THRESHOLD_XGB_INFO.get('optimized_threshold', 0.3103))
MAT_NN_THRESHOLD = 0.5

MAT_XGB_MODEL = None
MAT_NN_MODEL = None

try:
    if joblib is not None and os.path.exists(_XGB_MODEL_PATH):
        MAT_XGB_MODEL = joblib.load(_XGB_MODEL_PATH)
        print('[OK] Modelo XGBoost cargado')
except Exception as e:
    print(f'[!] No se pudo cargar el modelo XGBoost — {e}')

try:
    if keras is not None and os.path.exists(_NN_MODEL_PATH):
        MAT_NN_MODEL = keras.models.load_model(_NN_MODEL_PATH)
        print('[OK] Modelo de red neuronal cargado')
except Exception as e:
    print(f'[!] No se pudo cargar el modelo de red neuronal — {e}')

# ── v2 production models ──────────────────────────────────────────────
ING_V2_MODEL = ING_V2_PREP = ING_V2_META = None
REG_V2_MODEL = REG_V2_PREP = REG_V2_META = None

try:
    if joblib is not None and keras is not None:
        ING_V2_MODEL = keras.models.load_model(
            os.path.join(_MODELS_DIR, 'modelo_ingles_A-_v2.keras'))
        ING_V2_PREP = joblib.load(
            os.path.join(_MODELS_DIR, 'preprocessor_ingles_A-_v2.pkl'))
        ING_V2_META = joblib.load(
            os.path.join(_MODELS_DIR, 'metadata_ingles_A-_v2.pkl'))
        print('[OK] Modelo inglés A- v2 cargado')
except Exception as e:
    print(f'[!] Modelo inglés v2 — {e}')

try:
    if joblib is not None and keras is not None:
        REG_V2_MODEL = keras.models.load_model(
            os.path.join(_MODELS_DIR, 'modelo_puntaje_global_v2.keras'))
        REG_V2_PREP = joblib.load(
            os.path.join(_MODELS_DIR, 'preprocessor_puntaje_global_v2.pkl'))
        REG_V2_META = joblib.load(
            os.path.join(_MODELS_DIR, 'metadata_puntaje_global_v2.pkl'))
        print('[OK] Modelo puntaje global v2 cargado')
except Exception as e:
    print(f'[!] Modelo puntaje global v2 — {e}')

df = None
_hg = _hm = None  # histogram tuples
agg_year = agg_area = agg_nat = agg_ingles = agg_mcpio = None
school_by_mcpio = {}

try:
    df = pd.read_csv(_CSV, low_memory=False)
    for col in ('punt_global', 'punt_matematicas', 'periodo'):
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['periodo']).copy()
    df['year'] = (df['periodo'] // 10).astype(int)

    _g = df.dropna(subset=['punt_global'])
    agg_year = (_g.groupby('year')['punt_global']
                .agg(mean='mean', std='std', n='count').reset_index())
    agg_area = (_g.groupby('cole_area_ubicacion')['punt_global']
                .mean().reset_index().rename(columns={'punt_global': 'mean',
                                                      'cole_area_ubicacion': 'area'}))
    agg_nat  = (_g.groupby('cole_naturaleza')['punt_global']
                .mean().reset_index().rename(columns={'punt_global': 'mean',
                                                      'cole_naturaleza': 'nat'}))
    agg_mcpio = (_g.groupby('cole_mcpio_ubicacion')['punt_global']
                 .agg(mean='mean', n='count').reset_index().sort_values('mean').tail(25))

    _ing = df.dropna(subset=['desemp_ingles'])
    _ord = ['A-', 'A1', 'A2', 'B1', 'B+']
    agg_ingles = (_ing['desemp_ingles'].value_counts()
                  .reindex(_ord).reset_index())
    agg_ingles.columns = ['nivel', 'n']
    agg_ingles = agg_ingles.dropna()

    _hg = np.histogram(_g['punt_global'].values, bins=60, range=(50, 500))
    _m  = df.dropna(subset=['punt_matematicas'])
    _hm = np.histogram(_m['punt_matematicas'].values, bins=50, range=(20, 100))

    # school → municipality mapping for simulator
    school_by_mcpio = {}
    _sdf = df.dropna(subset=['cole_nombre_establecimiento', 'cole_mcpio_ubicacion'])
    for _mc, _grp in _sdf.groupby('cole_mcpio_ubicacion'):
        school_by_mcpio[_mc] = sorted(_grp['cole_nombre_establecimiento'].unique().tolist())

    print(f'[OK] CSV: {df.shape[0]:,} filas')
except Exception as _e:
    print(f'[!]  CSV no disponible — {_e}')
    school_by_mcpio = {}


# ─────────────────────────────────────────────────────────────────────
# 3. HARDCODED MODEL RESULTS
# ─────────────────────────────────────────────────────────────────────
REG_MODELS = pd.DataFrame([
    dict(label='Lasso',                  grp='Lineal',        rmse=38.665, mae=30.829, r2=0.2792),
    dict(label='NN-REF (todas)',          grp='NN sin Lasso',  rmse=38.137, mae=30.134, r2=0.2988),
    dict(label='NN-1 (64)',              grp='NN con Lasso',  rmse=37.558, mae=29.884, r2=0.3199),
    dict(label='NN-2 (128-64)',          grp='NN con Lasso',  rmse=37.841, mae=30.012, r2=0.3096),
    dict(label='NN-3 (64-32)',           grp='NN con Lasso',  rmse=38.026, mae=30.133, r2=0.3029),
    dict(label='NN-4 (128-64-32)',       grp='NN con Lasso',  rmse=38.320, mae=30.256, r2=0.2921),
    dict(label='NN-5 (256-128-64)',      grp='NN con Lasso',  rmse=37.999, mae=30.087, r2=0.3039),
    dict(label='NN-6 (128-64-32, lr↓)', grp='NN con Lasso',  rmse=38.362, mae=30.332, r2=0.2905),
])

LASSO_COEFS = pd.DataFrame([
    ('Calendario A vs. B',       -44.85),
    ('Jornada Sabatina',         -43.40),
    ('Jornada Nocturna',         -28.60),
    ('Colegio Oficial',          -21.40),
    ('Padre sin educación',      -18.50),
    ('Madre sin educación',      -15.80),
    ('Zona Rural',               -10.10),
    ('Estrato 1',                 -8.20),
    ('Padre con Posgrado',       +26.80),
    ('Colegio Privado',          +21.40),
    ('Madre con Posgrado',       +20.50),
    ('Jornada Completa',         +15.50),
    ('Estrato 6',                +15.20),
    ('Colegio Bilingüe',         +12.30),
    ('Computador en el hogar',   +10.20),
    ('Internet en el hogar',      +9.80),
], columns=['var', 'coef']).sort_values('coef')

ING_MODELS = pd.DataFrame([
    dict(label='Reg. Logística',             grp='Baseline',       f1=0.7213, auc=0.7037, pr=0.7597),
    dict(label='NN base (64-32)',             grp='NN v1',          f1=0.7488, auc=0.7083, pr=0.7668),
    dict(label='NN small (32-16)',            grp='NN v1',          f1=0.7450, auc=0.7088, pr=0.7670),
    dict(label='NN med (128-64)',             grp='NN v1',          f1=0.7450, auc=0.7086, pr=0.7674),
    dict(label='NN deep (128-64-32)',         grp='NN v1',          f1=0.7540, auc=0.7074, pr=0.7661),
    dict(label='NN lowlr (64-32)',            grp='NN v1',          f1=0.7363, auc=0.7088, pr=0.7673),
    dict(label='NN big (256-128-64)',         grp='NN v1',          f1=0.7364, auc=0.7081, pr=0.7661),
    dict(label='NN v2 (municipio + colegio)', grp='NN v2 (final)', f1=0.7926, auc=0.7350, pr=0.7871),
])

ING_IMPORTANCE = pd.DataFrame([
    ('Tasa histórica A- por colegio', 0.142),
    ('Año del período',               0.098),
    ('Municipio del colegio',         0.087),
    ('Jornada escolar',               0.065),
    ('Tipo de colegio (ofic./priv.)', 0.058),
    ('Internet en el hogar',          0.051),
    ('Computador en el hogar',        0.048),
    ('Educación del padre',           0.045),
    ('Estrato socioeconómico',        0.042),
    ('Educación de la madre',         0.040),
], columns=['feature', 'importance'])

ING_CM   = np.array([[3544, 7816], [920, 16831]])
REG_TEST = dict(rmse=38.088, mae=30.352, r2=0.3153, mape=12.93)
ING_TEST = dict(f1=0.794, auc=0.740, pr=0.792, recall=0.948, precision=0.683)
MAT = dict(
    n_raw=100094,
    n_clean=29714,
    n_test=5943,
    n_pos_test=235,
    pct_cumple=3.95,
    pct_no=96.05,
    threshold_score=70
)

MAT_MODELS = {
    'xgb': {
        'label': 'Modelo XGBoost',
        'short': 'XGBoost',
        'threshold': 0.3103,
        'accuracy': 0.7799,
        'recall': 0.8255,
        'f1': 0.2288,
        'auc': 0.8856,
        'cm': np.array([[4441, 1267], [41, 194]]),
        'note': 'Modelo seleccionado: mayor recall y AUC en test. Reduce los falsos negativos frente a la red neuronal.'
    },
    'nn': {
        'label': 'Modelo de Redes Neuronales',
        'short': 'Red neuronal',
        'threshold': 0.5,
        'accuracy': 0.7871,
        'recall': 0.7660,
        'f1': 0.2215,
        'auc': 0.8743,
        'cm': np.array([[4498, 1210], [55, 180]]),
        'note': 'Modelo base de red neuronal con buen AUC, pero menor recall que XGBoost para la clase positiva.'
    }
}

MAT_FEATURES = [
    'anio_periodo',
    'periodo_aplicacion',

    'cole_area_ubicacion',
    'cole_bilingue',
    'cole_calendario',
    'cole_caracter',
    'cole_genero',
    'cole_jornada',
    'cole_naturaleza',
    'cole_sede_principal',
    'cole_mcpio_ubicacion',

    'edad_aprox',
    'estu_genero',
    'estu_mcpio_reside',

    'fami_cuartoshogar',
    'fami_educacionmadre',
    'fami_educacionpadre',
    'fami_estratovivienda',
    'fami_personashogar',
    'fami_tieneautomovil',
    'fami_tienecomputador',
    'fami_tieneinternet',
    'fami_tienelavadora'
]

MAT_NUMERIC_FEATURES = [
    'anio_periodo',
    'periodo_aplicacion',
    'edad_aprox'
]

MAT_FEATURE_LABELS = {
    'anio_periodo': 'Año en el que se toma el examen',
    'periodo_aplicacion': 'Periodo de aplicación del examen',

    'cole_area_ubicacion': 'Área de ubicación del colegio',
    'cole_bilingue': 'Colegio bilingüe',
    'cole_calendario': 'Calendario del colegio',
    'cole_caracter': 'Carácter del colegio',
    'cole_genero': 'Género del colegio',
    'cole_jornada': 'Jornada del colegio',
    'cole_naturaleza': 'Naturaleza del colegio',
    'cole_sede_principal': '¿Es sede principal?',
    'cole_mcpio_ubicacion': 'Municipio del colegio',

    'edad_aprox': 'Edad aproximada del estudiante',
    'estu_genero': 'Género del estudiante',
    'estu_mcpio_reside': 'Municipio de residencia del estudiante',

    'fami_cuartoshogar': 'Número de cuartos en el hogar',
    'fami_educacionmadre': 'Nivel educativo de la madre',
    'fami_educacionpadre': 'Nivel educativo del padre',
    'fami_estratovivienda': 'Estrato de la vivienda',
    'fami_personashogar': 'Número de personas en el hogar',
    'fami_tieneautomovil': '¿Tiene automóvil en el hogar?',
    'fami_tienecomputador': '¿Tiene computador en el hogar?',
    'fami_tieneinternet': '¿Tiene internet en el hogar?',
    'fami_tienelavadora': '¿Tiene lavadora en el hogar?'
}

MAT_FEATURE_GROUPS = [
    ('Información del examen', [
        'anio_periodo',
        'periodo_aplicacion'
    ]),
    ('Información del colegio', [
        'cole_area_ubicacion',
        'cole_bilingue',
        'cole_calendario',
        'cole_caracter',
        'cole_genero',
        'cole_jornada',
        'cole_naturaleza',
        'cole_sede_principal',
        'cole_mcpio_ubicacion'
    ]),
    ('Información del estudiante', [
        'edad_aprox',
        'estu_genero',
        'estu_mcpio_reside'
    ]),
    ('Información familiar y socioeconómica', [
        'fami_cuartoshogar',
        'fami_educacionmadre',
        'fami_educacionpadre',
        'fami_estratovivienda',
        'fami_personashogar',
        'fami_tieneautomovil',
        'fami_tienecomputador',
        'fami_tieneinternet',
        'fami_tienelavadora'
    ])
]

# ── Simulator dropdown options (exact values seen by preprocessor) ─────
SIM_OPTIONS = {
    'cole_area_ubicacion':  ['RURAL', 'URBANO'],
    'cole_naturaleza':      ['NO OFICIAL', 'OFICIAL'],
    'cole_jornada':         ['COMPLETA', 'MAÑANA', 'NOCHE', 'SABATINA', 'TARDE', 'UNICA'],
    'cole_calendario':      ['A', 'B'],
    'cole_caracter':        ['ACADÉMICO', 'NO APLICA', 'TÉCNICO', 'TÉCNICO/ACADÉMICO'],
    'cole_bilingue':        ['N', 'S'],
    'cole_genero':          ['FEMENINO', 'MIXTO'],
    'estu_genero':          ['F', 'M'],
    'fami_tieneinternet':   ['NO', 'SI'],
    'fami_tienecomputador': ['NO', 'SI'],
    'fami_tieneautomovil':  ['NO', 'SI'],
    'fami_tienelavadora':   ['NO', 'SI'],
    'fami_estratovivienda': ['ESTRATO 1', 'ESTRATO 2', 'ESTRATO 3',
                             'ESTRATO 4', 'ESTRATO 5', 'ESTRATO 6', 'SIN ESTRATO'],
    'fami_educacionmadre':  [
        'NINGUNO', 'PRIMARIA INCOMPLETA', 'PRIMARIA COMPLETA',
        'SECUNDARIA (BACHILLERATO) INCOMPLETA', 'SECUNDARIA (BACHILLERATO) COMPLETA',
        'TÉCNICA O TECNOLÓGICA INCOMPLETA', 'TÉCNICA O TECNOLÓGICA COMPLETA',
        'EDUCACIÓN PROFESIONAL INCOMPLETA', 'EDUCACIÓN PROFESIONAL COMPLETA',
        'POSTGRADO', 'NO APLICA', 'NO SABE',
    ],
    'fami_educacionpadre':  [
        'NINGUNO', 'PRIMARIA INCOMPLETA', 'PRIMARIA COMPLETA',
        'SECUNDARIA (BACHILLERATO) INCOMPLETA', 'SECUNDARIA (BACHILLERATO) COMPLETA',
        'TÉCNICA O TECNOLÓGICA INCOMPLETA', 'TÉCNICA O TECNOLÓGICA COMPLETA',
        'EDUCACIÓN PROFESIONAL INCOMPLETA', 'EDUCACIÓN PROFESIONAL COMPLETA',
        'POSTGRADO', 'NO APLICA', 'NO SABE',
    ],
    'fami_cuartoshogar':    ['UNO', 'DOS', 'TRES', 'CUATRO', 'CINCO',
                             'SEIS', 'SIETE', 'OCHO', 'NUEVE', 'DIEZ O MÁS'],
    'fami_personashogar':   ['1 A 2', '3 A 4', '5 A 6', '7 A 8', '9 O MÁS'],
}


# ─────────────────────────────────────────────────────────────────────
# 4. COMPONENT HELPERS
# ─────────────────────────────────────────────────────────────────────
def kpi(label, value, sub=None, color=BLUE):
    return html.Div([
        html.Div(label, className='metric-label'),
        html.Div(value, className='metric-value'),
        html.Div(sub or '', className='metric-sub'),
    ], className='metric-card', style={'borderTop': f'3px solid {color}'})


def chart_card(title, sub, fig, height=340):
    return html.Div([
        html.Div([
            html.Div(title, className='chart-title'),
            html.Div(sub,   className='chart-sub'),
        ], className='chart-card-header'),
        dcc.Graph(figure=fig, config=_CFG, style={'height': f'{height}px'}),
    ], className='chart-card')


def insight(content, kind='info'):
    cls = 'insight-box' + (' warn' if kind == 'warn' else
                            ' success' if kind == 'success' else '')
    return html.Div(content, className=cls)


def section(title, desc):
    return html.Div([
        html.H2(title, className='section-title'),
        html.P(desc,   className='section-desc'),
    ], className='section-header')


def metric_radio(label, id_, options, value):
    return html.Div([
        html.Span(label, className='selector-label'),
        dcc.RadioItems(
            id=id_,
            options=[{'label': o['l'], 'value': o['v']} for o in options],
            value=value,
            inline=True,
            inputStyle={'marginRight': '4px'},
            labelStyle={'marginRight': '16px', 'fontSize': '13px',
                        'color': SUB, 'cursor': 'pointer'},
        ),
    ], className='selector-row')


# ─────────────────────────────────────────────────────────────────────
# 5. CHART BUILDERS
# ─────────────────────────────────────────────────────────────────────

def _score_dist():
    if _hg:
        counts, edges = _hg
        mids = (edges[:-1] + edges[1:]) / 2
    else:
        mids   = np.linspace(50, 500, 60)
        counts = np.round(np.exp(-0.5 * ((mids - 241) / 46) ** 2)
                          * 118523 * (450 / 60) / (46 * np.sqrt(2 * np.pi))).astype(int)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=mids, y=counts,
        marker_color=MID, marker_line_width=0,
        hovertemplate='Puntaje %{x:.0f}<br>%{y:,} estudiantes<extra></extra>',
    ))
    for val, lbl, col, pos in [
        (241, 'Media 241',   RED, 'top right'),
        (235, 'Mediana 235', GRN, 'top left'),
    ]:
        fig.add_vline(x=val, line_dash='dash', line_color=col, line_width=2,
                      annotation_text=lbl, annotation_font_size=11,
                      annotation_font_color=col, annotation_position=pos)
    fig.update_layout(**_L, showlegend=False, bargap=0.02)
    fig.update_xaxes(**_AX, title='Puntaje global')
    fig.update_yaxes(**_AX, title='Estudiantes')
    return fig


def _score_trend():
    if agg_year is not None:
        yrs  = agg_year['year'].values
        mns  = agg_year['mean'].values
        stds = agg_year['std'].values
    else:
        yrs  = np.arange(2011, 2023)
        mns  = np.array([248, 250, 249, 247, 245, 243, 241, 240, 239, 240, 241, 242], dtype=float)
        stds = np.full(len(yrs), 45.0)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=np.concatenate([yrs, yrs[::-1]]),
        y=np.concatenate([mns + stds, (mns - stds)[::-1]]),
        fill='toself', fillcolor='rgba(91,155,213,0.12)',
        line=dict(width=0), showlegend=False, hoverinfo='skip',
    ))
    fig.add_trace(go.Scatter(
        x=yrs, y=mns, mode='lines+markers',
        line=dict(color=BLUE, width=2.5),
        marker=dict(size=8, color=BLUE, line=dict(color='white', width=2)),
        name='Promedio departamental',
        hovertemplate='Año %{x}<br>Puntaje promedio: %{y:.1f} pts<extra></extra>',
    ))
    fig.update_layout(**_L,
        xaxis=dict(title='Año', dtick=1, gridcolor='#efefef', linecolor='#efefef'),
        yaxis=dict(title='Puntaje promedio', gridcolor='#efefef', linecolor='#efefef'),
        showlegend=False)
    return fig


def _ingles_levels():
    if agg_ingles is not None:
        labels = agg_ingles['nivel'].tolist()
        values = agg_ingles['n'].tolist()
    else:
        labels = ['A-', 'A1', 'A2', 'B1', 'B+']
        values = [118336, 43210, 19853, 9254, 3417]

    total  = sum(values)
    pcts   = [v / total * 100 for v in values]
    colors = ['#b03a2e', '#d4651c', '#c9a418', '#3a7d44', '#2e6da4']

    fig = go.Figure(go.Bar(
        x=labels, y=values,
        marker_color=colors, marker_line_width=0,
        text=[f'{p:.1f}%' for p in pcts],
        textposition='outside',
        hovertemplate='Nivel %{x}<br>%{y:,} estudiantes (%{text})<extra></extra>',
    ))
    fig.update_layout(**_L,
        xaxis=dict(title='Nivel Marco Europeo', gridcolor='rgba(0,0,0,0)',
                   linecolor='#efefef'),
        yaxis=dict(title='Estudiantes', gridcolor='#efefef', linecolor='#efefef'),
        showlegend=False)
    return fig


def _score_by_area():
    if agg_area is not None and agg_nat is not None:
        areas   = agg_area['area'].tolist()
        a_means = agg_area['mean'].tolist()
        nats    = agg_nat['nat'].tolist()
        n_means = agg_nat['mean'].tolist()
    else:
        areas   = ['RURAL', 'URBANO']
        a_means = [228.4, 248.2]
        nats    = ['OFICIAL', 'NO OFICIAL']
        n_means = [231.6, 269.4]

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=['Área de ubicación', 'Naturaleza del colegio'],
                        horizontal_spacing=0.12)

    a_colors = [RED if 'RURAL' in a.upper() else BLUE for a in areas]
    n_colors = [GRN if 'NO OFICIAL' in n.upper() else LGRAY for n in nats]

    fig.add_trace(go.Bar(
        x=areas, y=a_means, marker_color=a_colors, marker_line_width=0,
        text=[f'{v:.1f}' for v in a_means], textposition='outside',
        hovertemplate='%{x}<br>Promedio: %{y:.1f} pts<extra></extra>',
    ), row=1, col=1)
    fig.add_trace(go.Bar(
        x=nats, y=n_means, marker_color=n_colors, marker_line_width=0,
        text=[f'{v:.1f}' for v in n_means], textposition='outside',
        hovertemplate='%{x}<br>Promedio: %{y:.1f} pts<extra></extra>',
    ), row=1, col=2)

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=_L['font'], margin=_L['margin'], showlegend=False,
    )
    fig.update_annotations(font=dict(size=12, color=TEXT))
    fig.update_yaxes(gridcolor='#efefef', linecolor='#efefef', range=[200, 295])
    fig.update_xaxes(gridcolor='rgba(0,0,0,0)', linecolor='rgba(0,0,0,0)')
    return fig


def _mcpio_scores():
    if agg_mcpio is not None:
        mcps = agg_mcpio['cole_mcpio_ubicacion'].tolist()
        mns  = agg_mcpio['mean'].tolist()
    else:
        mcps = ['Montería', 'Tierralta', 'Lorica', 'Cereté', 'Sahagún',
                'Ciénaga de Oro', 'Montelíbano', 'Planeta Rica']
        mns  = [258.4, 242.1, 240.8, 238.9, 237.2, 235.6, 233.1, 231.4]

    n = len(mcps)
    colors = [NAVY if i >= n - 3 else BLUE if i >= n - 8 else '#a8c4e0'
              for i in range(n)]

    fig = go.Figure(go.Bar(
        x=mns, y=mcps,
        orientation='h',
        marker_color=colors, marker_line_width=0,
        text=[f'{v:.1f}' for v in mns], textposition='outside',
        textfont=dict(size=11),
        hovertemplate='%{y}<br>Puntaje promedio: %{x:.1f} pts<extra></extra>',
    ))
    fig.update_layout(**_L,
        xaxis=dict(title='Puntaje promedio', gridcolor='#efefef', linecolor='#efefef',
                   range=[min(mns) * 0.96, max(mns) * 1.06]),
        yaxis=dict(gridcolor='rgba(0,0,0,0)', linecolor='rgba(0,0,0,0)',
                   tickfont=dict(size=11)),
        showlegend=False)
    return fig


def _lasso_coefs():
    colors = [RED if c < 0 else BLUE for c in LASSO_COEFS['coef']]
    fig = go.Figure(go.Bar(
        x=LASSO_COEFS['coef'], y=LASSO_COEFS['var'],
        orientation='h',
        marker_color=colors, marker_line_width=0,
        text=[f'{v:+.1f}' for v in LASSO_COEFS['coef']],
        textposition='outside', textfont=dict(size=11),
        hovertemplate='%{y}<br>Coeficiente: %{x:+.2f} pts<extra></extra>',
    ))
    fig.add_vline(x=0, line_color='#374151', line_width=1)
    fig.update_layout(**_L,
        xaxis=dict(title='Efecto sobre el puntaje (pts)',
                   gridcolor='#efefef', linecolor='#efefef', zeroline=False),
        yaxis=dict(gridcolor='rgba(0,0,0,0)', linecolor='rgba(0,0,0,0)',
                   tickfont=dict(size=11)),
        showlegend=False)
    return fig


def _reg_comparison(metric='rmse'):
    col_map = {'rmse': ('RMSE (pts)', True), 'mae': ('MAE (pts)', True), 'r2': ('R²', False)}
    col_name, lower = col_map[metric]
    best = REG_MODELS[metric].idxmin() if lower else REG_MODELS[metric].idxmax()

    colors = []
    for i, row in REG_MODELS.iterrows():
        if i == best:
            colors.append('#c0392b')
        elif row['grp'] == 'Lineal':
            colors.append('#c4d4e6')
        elif row['grp'] == 'NN sin Lasso':
            colors.append(MID)
        else:
            colors.append(BLUE)

    baseline = REG_MODELS.loc[REG_MODELS['grp'] == 'Lineal', metric].values[0]
    ymin, ymax = REG_MODELS[metric].min(), REG_MODELS[metric].max()
    yrange = [0, ymax * 1.18] if metric == 'r2' else [ymin * 0.97, ymax * 1.065]

    fig = go.Figure(go.Bar(
        x=REG_MODELS['label'], y=REG_MODELS[metric],
        marker_color=colors, marker_line_width=0,
        text=[f'{v:.3f}' for v in REG_MODELS[metric]],
        textposition='outside', textfont=dict(size=10),
        hovertemplate='<b>%{x}</b><br>' + col_name + ': %{y:.4f}<extra></extra>',
    ))
    fig.add_hline(y=baseline, line_dash='dot', line_color='#bbb', line_width=1.5,
                  annotation_text='Lasso base', annotation_font_size=10,
                  annotation_font_color='#999', annotation_position='top right')
    fig.update_layout(**_L,
        xaxis=dict(tickangle=-28, gridcolor='rgba(0,0,0,0)',
                   linecolor='#efefef', tickfont=dict(size=10)),
        yaxis=dict(title=col_name, gridcolor='#efefef', linecolor='#efefef',
                   range=yrange),
        showlegend=False)
    return fig


def _ing_comparison(metric='f1'):
    col_map = {'f1': 'F1-Score', 'auc': 'ROC-AUC', 'pr': 'PR-AUC'}
    col_name = col_map[metric]
    best = ING_MODELS[metric].idxmax()

    colors = []
    for i, row in ING_MODELS.iterrows():
        if i == best:
            colors.append('#c0392b')
        elif row['grp'] == 'Baseline':
            colors.append('#c4d4e6')
        elif row['grp'] == 'NN v2 (final)':
            colors.append(NAVY)
        else:
            colors.append(MID)

    baseline = ING_MODELS.loc[ING_MODELS['grp'] == 'Baseline', metric].values[0]
    ymin, ymax = ING_MODELS[metric].min(), ING_MODELS[metric].max()

    fig = go.Figure(go.Bar(
        x=ING_MODELS['label'], y=ING_MODELS[metric],
        marker_color=colors, marker_line_width=0,
        text=[f'{v:.3f}' for v in ING_MODELS[metric]],
        textposition='outside', textfont=dict(size=10),
        hovertemplate='<b>%{x}</b><br>' + col_name + ': %{y:.4f}<extra></extra>',
    ))
    fig.add_hline(y=baseline, line_dash='dot', line_color='#bbb', line_width=1.5,
                  annotation_text='Baseline logístico', annotation_font_size=10,
                  annotation_font_color='#999', annotation_position='top right')
    fig.update_layout(**_L,
        xaxis=dict(tickangle=-28, gridcolor='rgba(0,0,0,0)',
                   linecolor='#efefef', tickfont=dict(size=10)),
        yaxis=dict(title=col_name, gridcolor='#efefef', linecolor='#efefef',
                   range=[ymin * 0.97, ymax * 1.07]),
        showlegend=False)
    return fig


def _confusion_matrix():
    cm    = ING_CM
    total = cm.sum()
    z     = cm / total * 100

    annots = []
    for i in range(2):
        for j in range(2):
            fc = 'white' if z[i, j] > 20 else '#1a1a2e'
            annots.append(dict(
                x=j, y=i,
                text=f'<b>{cm[i,j]:,}</b><br>{z[i,j]:.1f}%',
                showarrow=False,
                font=dict(size=14, color=fc),
                xref='x', yref='y',
            ))
    cell_labels = ['Verdaderos Negativos', 'Falsos Positivos',
                   'Falsos Negativos',     'Verdaderos Positivos']
    for idx, lbl in enumerate(cell_labels):
        i, j = divmod(idx, 2)
        fc = 'rgba(255,255,255,0.65)' if z[i, j] > 20 else 'rgba(30,58,95,0.45)'
        annots.append(dict(
            x=j, y=i + 0.32,
            text=f'<i style="font-size:11px">{lbl}</i>',
            showarrow=False,
            font=dict(size=10, color=fc),
            xref='x', yref='y',
        ))

    fig = go.Figure(go.Heatmap(
        z=z,
        x=['Predicho: No A-', 'Predicho: A-'],
        y=['Real: No A-', 'Real: A-'],
        colorscale=[[0, '#eef4fc'], [0.35, '#5b9bd5'], [1, '#1e3a5f']],
        showscale=False,
        hovertemplate='%{y} / %{x}<br>%{z:.1f}% del total<extra></extra>',
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=_L['font'], margin=dict(l=10, r=10, t=14, b=10),
        annotations=annots,
        xaxis=dict(side='top', gridcolor='rgba(0,0,0,0)', linecolor='rgba(0,0,0,0)'),
        yaxis=dict(autorange='reversed', gridcolor='rgba(0,0,0,0)',
                   linecolor='rgba(0,0,0,0)'),
    )
    return fig


def _feature_importance():
    fig = go.Figure(go.Bar(
        x=ING_IMPORTANCE['importance'],
        y=ING_IMPORTANCE['feature'],
        orientation='h',
        marker_color=[NAVY if i == 0 else BLUE if i < 3 else MID
                      for i in range(len(ING_IMPORTANCE))],
        marker_line_width=0,
        text=[f'{v:.3f}' for v in ING_IMPORTANCE['importance']],
        textposition='outside', textfont=dict(size=11),
        hovertemplate='%{y}<br>Importancia: %{x:.3f}<extra></extra>',
    ))
    fig.update_layout(**_L,
        xaxis=dict(title='Importancia relativa', gridcolor='#efefef', linecolor='#efefef'),
        yaxis=dict(gridcolor='rgba(0,0,0,0)', linecolor='rgba(0,0,0,0)',
                   tickfont=dict(size=11), autorange='reversed'),
        showlegend=False)
    return fig


def _mat_donut():
    fig = go.Figure(go.Pie(
        labels=['No cumple (≤ 70)', 'Cumple (> 70)'],
        values=[MAT['pct_no'], MAT['pct_cumple']],
        hole=0.64,
        marker=dict(colors=['#d1d9e6', BLUE],
                    line=dict(color='white', width=2)),
        textinfo='percent+label',
        textfont=dict(size=12),
        hovertemplate='%{label}<br>%{percent}<extra></extra>',
        sort=False,
    ))
    fig.add_annotation(
        text=f'<b>{MAT["pct_no"]:.1f}%</b><br><span style="font-size:11px">no cumple</span>',
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=16, color=TEXT), align='center',
    )
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=_L['font'], margin=dict(l=10, r=10, t=20, b=10),
        showlegend=True,
        legend=dict(orientation='h', x=0.5, xanchor='center', y=-0.08,
                    bgcolor='rgba(0,0,0,0)', font=dict(size=12)),
    )
    return fig


def _mat_dist():
    if _hm:
        counts, edges = _hm
        mids = (edges[:-1] + edges[1:]) / 2
    else:
        mids   = np.linspace(20, 100, 50)
        counts = np.round(np.exp(-0.5 * ((mids - 50) / 14) ** 2)
                          * 80000 * 80 / 50 / (14 * np.sqrt(2 * np.pi))).astype(int)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=mids, y=counts,
        marker_color=[RED if m <= 70 else GRN for m in mids],
        marker_line_width=0,
        hovertemplate='Puntaje %{x:.0f}<br>%{y:,} estudiantes<extra></extra>',
    ))
    fig.add_vline(x=70, line_dash='dash', line_color=NAVY, line_width=2,
                  annotation_text='Umbral = 70', annotation_font_size=12,
                  annotation_font_color=NAVY, annotation_position='top right')
    fig.update_layout(**_L, showlegend=False, bargap=0.02)
    fig.update_xaxes(**_AX, title='Puntaje de matemáticas')
    fig.update_yaxes(**_AX, title='Estudiantes')
    return fig

def _mat_model_data(model_key='xgb'):
    return MAT_MODELS.get(model_key or 'xgb', MAT_MODELS['xgb'])


def _mat_kpi_cards(model_key='xgb'):
    m = _mat_model_data(model_key)

    return [
        kpi('Accuracy — test', f'{m["accuracy"]:.3f}',
            'Exactitud global del modelo', NAVY),
        kpi('Recall — test', f'{m["recall"]:.1%}',
            'Métrica principal: detecta estudiantes que sí cumplen', BLUE),
        kpi('F1-score — test', f'{m["f1"]:.3f}',
            'Balance entre precisión y recall', GRN),
        kpi('ROC-AUC — test', f'{m["auc"]:.3f}',
            'Capacidad discriminativa del modelo', MID),
        kpi('Umbral usado', f'{m["threshold"]:.3f}',
            'Punto de corte para clasificar cumple/no cumple', AMB),
    ]


def _mat_confusion_matrix_model(model_key='xgb'):
    m = _mat_model_data(model_key)
    cm = m['cm']
    total = cm.sum()
    z = cm / total * 100

    annots = []
    for i in range(2):
        for j in range(2):
            fc = 'white' if z[i, j] > 20 else '#1a1a2e'
            annots.append(dict(
                x=j, y=i,
                text=f'<b>{cm[i,j]:,}</b><br>{z[i,j]:.1f}%',
                showarrow=False,
                font=dict(size=14, color=fc),
                xref='x', yref='y',
            ))

    cell_labels = [
        'Verdaderos Negativos',
        'Falsos Positivos',
        'Falsos Negativos',
        'Verdaderos Positivos'
    ]

    for idx, lbl in enumerate(cell_labels):
        i, j = divmod(idx, 2)
        fc = 'rgba(255,255,255,0.70)' if z[i, j] > 20 else 'rgba(30,58,95,0.50)'
        annots.append(dict(
            x=j, y=i + 0.32,
            text=f'<i style="font-size:11px">{lbl}</i>',
            showarrow=False,
            font=dict(size=10, color=fc),
            xref='x', yref='y',
        ))

    fig = go.Figure(go.Heatmap(
        z=z,
        x=['Predicho: No cumple', 'Predicho: Cumple'],
        y=['Real: No cumple', 'Real: Cumple'],
        colorscale=[[0, '#eef4fc'], [0.35, '#5b9bd5'], [1, '#1e3a5f']],
        showscale=False,
        hovertemplate='%{y} / %{x}<br>%{z:.1f}% del total<extra></extra>',
    ))

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=_L['font'],
        margin=dict(l=10, r=10, t=14, b=10),
        annotations=annots,
        xaxis=dict(side='top', gridcolor='rgba(0,0,0,0)',
                   linecolor='rgba(0,0,0,0)'),
        yaxis=dict(autorange='reversed', gridcolor='rgba(0,0,0,0)',
                   linecolor='rgba(0,0,0,0)'),
    )

    return fig


def _mat_metrics_bar(model_key='xgb'):
    m = _mat_model_data(model_key)

    labels = ['Accuracy', 'Recall', 'F1-score', 'ROC-AUC']
    values = [m['accuracy'], m['recall'], m['f1'], m['auc']]
    colors = [NAVY, BLUE, GRN, MID]

    fig = go.Figure(go.Bar(
        x=labels,
        y=values,
        marker_color=colors,
        marker_line_width=0,
        text=[f'{v:.3f}' for v in values],
        textposition='outside',
        textfont=dict(size=11),
        hovertemplate='%{x}: %{y:.4f}<extra></extra>',
    ))

    fig.update_layout(
        **_L,
        yaxis=dict(title='Valor de la métrica',
                   gridcolor='#efefef',
                   linecolor='#efefef',
                   range=[0, 1]),
        xaxis=dict(gridcolor='rgba(0,0,0,0)',
                   linecolor='#efefef'),
        showlegend=False
    )

    return fig


def _mat_input_field(col):
    label = MAT_FEATURE_LABELS.get(col, col)

    common_style = {
        'minWidth': '230px',
        'flex': '1 1 230px',
        'marginBottom': '12px'
    }

    label_style = {
        'fontSize': '12px',
        'fontWeight': '600',
        'color': TEXT,
        'display': 'block',
        'marginBottom': '6px'
    }

    if col in MAT_NUMERIC_FEATURES:
        info = CATEGORIAS_MODELO.get('numeric_features', {}).get(col, {})
        median = info.get('median', 0)
        min_v = info.get('min', None)
        max_v = info.get('max', None)

        if col == 'anio_periodo' and not median:
            median = 2022
        if col == 'periodo_aplicacion' and not median:
            median = 2
        if col == 'edad_aprox' and not median:
            median = 17

        return html.Div([
            html.Label(label, style=label_style),
            dcc.Input(
                id=f'mat-input-{col}',
                type='number',
                value=round(float(median), 0),
                min=min_v,
                max=max_v,
                debounce=True,
                style={
                    'width': '100%',
                    'padding': '10px 12px',
                    'border': f'1px solid {BORD}',
                    'borderRadius': '10px',
                    'fontSize': '13px'
                }
            )
        ], style=common_style)

    values = CATEGORIAS_MODELO.get('categorical_features', {}).get(col, [])

    options = [
        {'label': str(v), 'value': str(v)}
        for v in values
        if pd.notna(v) and str(v).strip() != ''
    ]

    default_value = options[0]['value'] if options else None

    return html.Div([
        html.Label(label, style=label_style),
        dcc.Dropdown(
            id=f'mat-input-{col}',
            options=options,
            value=default_value,
            clearable=False,
            placeholder='Seleccione una opción',
            style={
                'fontSize': '13px'
            }
        )
    ], style=common_style)


def _mat_input_section(title, cols):
    return html.Div([
        html.Div(title, className='chart-title',
                 style={'marginBottom': '14px'}),
        html.Div([
            _mat_input_field(col) for col in cols
        ], style={
            'display': 'flex',
            'flexWrap': 'wrap',
            'gap': '12px'
        })
    ], style={
        'borderTop': f'1px solid {BORD}',
        'paddingTop': '18px',
        'marginTop': '18px'
    })

# ─────────────────────────────────────────────────────────────────────
# 5b. SIMULATOR HELPERS
# ─────────────────────────────────────────────────────────────────────

def _predict_both(school_name, mcpio, area, naturaleza, jornada, calendario,
                  bilingue, caracter, cole_genero, estu_genero, estrato,
                  educ_madre, educ_padre, internet, computador, automovil,
                  lavadora, cuartos, personas, edad):
    base = {
        'cole_area_ubicacion':  area,
        'cole_naturaleza':      naturaleza,
        'cole_jornada':         jornada,
        'cole_calendario':      calendario,
        'cole_caracter':        caracter,
        'cole_bilingue':        bilingue,
        'cole_genero':          cole_genero,
        'estu_genero':          estu_genero,
        'fami_tieneinternet':   internet,
        'fami_tienecomputador': computador,
        'fami_tieneautomovil':  automovil,
        'fami_tienelavadora':   lavadora,
        'cole_mcpio_ubicacion': mcpio,
        'fami_estratovivienda': estrato,
        'fami_educacionmadre':  educ_madre,
        'fami_educacionpadre':  educ_padre,
        'fami_cuartoshogar':    cuartos,
        'fami_personashogar':   personas,
        'edad': float(edad),
    }

    enc_ing = ING_V2_META['school_encoding'].get(
        school_name, ING_V2_META['p_global'])
    row_ing = {**base, 'colegio_enc': enc_ing}
    X_ing   = ING_V2_PREP.transform(pd.DataFrame([row_ing]))
    prob_ing = float(ING_V2_MODEL.predict(X_ing, verbose=0)[0][0])

    enc_reg = REG_V2_META['school_encoding'].get(
        school_name, REG_V2_META['p_global'])
    row_reg = {**base, 'colegio_enc': enc_reg}
    X_reg   = REG_V2_PREP.transform(pd.DataFrame([row_reg]))
    pred_global = float(REG_V2_MODEL.predict(X_reg, verbose=0)[0][0])

    return prob_ing, pred_global


def _sim_dd(id_, opts, val):
    return dcc.Dropdown(
        id=id_,
        options=[{'label': o, 'value': o} for o in opts],
        value=val, clearable=False, style={'fontSize': '13px'})


def _sim_field(label_, child):
    return html.Div([
        html.Div(label_, style={'fontSize': '12px', 'fontWeight': '600',
                                'color': TEXT, 'marginBottom': '6px'}),
        child,
    ], style={'flex': '1 1 190px', 'minWidth': '190px'})


def _tab_simulador():
    if school_by_mcpio:
        mcpio_list = sorted(school_by_mcpio.keys())
    else:
        mcpio_list = [
            'AYAPEL', 'BUENAVISTA', 'CANALETE', 'CERETÉ', 'CHIMÁ', 'CHINÚ',
            'CIÉNAGA DE ORO', 'COTORRA', 'LA APARTADA', 'LORICA', 'LOS CÓRDOBAS',
            'MOMIL', 'MONTELÍBANO', 'MONTERÍA', 'MOÑITOS', 'PLANETA RICA',
            'PUEBLO NUEVO', 'PUERTO ESCONDIDO', 'PUERTO LIBERTADOR', 'PURÍSIMA',
            'SAHAGÚN', 'SAN ANDRÉS DE SOTAVENTO', 'SAN ANTERO',
            'SAN BERNARDO DEL VIENTO', 'SAN CARLOS', 'SAN JOSÉ DE URÉ',
            'SAN PELAYO', 'TIERRALTA', 'TUCHÍN', 'VALENCIA',
        ]
    mcpio_opts   = [{'label': m, 'value': m} for m in mcpio_list]
    default_mcpio = mcpio_list[0]
    first_schools = school_by_mcpio.get(default_mcpio, [])
    school_opts   = [{'label': s, 'value': s} for s in first_schools]
    default_school = first_schools[0] if first_schools else None

    models_ok = ING_V2_MODEL is not None and REG_V2_MODEL is not None
    status_col = GRN if models_ok else '#d97706'
    status_lbl = 'Modelos cargados y listos' if models_ok else 'Modelos no disponibles'

    return html.Div([
        section(
            'Simulador de Predicción Individual',
            'Estime el puntaje global Saber 11 y la probabilidad de nivel A- en inglés '
            'para cualquier perfil de estudiante del departamento de Córdoba. '
            'Los modelos son redes neuronales entrenadas sobre 194.259 registros 2011–2022. '
            'Los resultados son orientativos y deben interpretarse con el intervalo ±RMSE.',
        ),

        html.Div([
            kpi('Modelo puntaje global', 'NN v2',
                'R² = 0.378 · RMSE = 36.3 pts · encoding de colegio', BLUE),
            kpi('Modelo inglés A-', 'NN v2',
                'F1 = 0.794 · AUC = 0.740 · recall A- = 94.8%', GRN),
            kpi('Media departamental', '241 pts',
                'Referencia histórica del puntaje global', NAVY),
            kpi('Umbral inglés A-', '0.33',
                'Calibrado para maximizar F1 en validación', MID),
            html.Div([
                html.Div('Estado del simulador', className='metric-label'),
                html.Div(
                    html.Span(status_lbl, className='summary-card-badge',
                              style={'background': '#f0fdf4' if models_ok else '#fef3c7',
                                     'color': status_col}),
                    style={'marginTop': '6px'}
                ),
            ], className='metric-card',
               style={'borderTop': f'3px solid {status_col}'}),
        ], className='metrics-row'),

        html.Div([
            html.Div('Perfil del estudiante', className='chart-title',
                     style={'marginBottom': '18px'}),

            html.Div([
                _sim_field('Municipio del colegio',
                           dcc.Dropdown(id='sim-mcpio', options=mcpio_opts,
                                        value=default_mcpio, clearable=False,
                                        style={'fontSize': '13px'})),
                _sim_field('Colegio',
                           dcc.Dropdown(id='sim-school', options=school_opts,
                                        value=default_school, clearable=False,
                                        placeholder='Seleccione municipio primero',
                                        style={'fontSize': '13px'})),
                _sim_field('Área', _sim_dd('sim-area',
                           SIM_OPTIONS['cole_area_ubicacion'], 'URBANO')),
                _sim_field('Naturaleza', _sim_dd('sim-naturaleza',
                           SIM_OPTIONS['cole_naturaleza'], 'OFICIAL')),
                _sim_field('Jornada', _sim_dd('sim-jornada',
                           SIM_OPTIONS['cole_jornada'], 'MAÑANA')),
            ], style={'display': 'flex', 'flexWrap': 'wrap', 'gap': '14px',
                      'marginBottom': '16px'}),

            html.Div([
                _sim_field('Calendario', _sim_dd('sim-calendario',
                           SIM_OPTIONS['cole_calendario'], 'A')),
                _sim_field('Carácter', _sim_dd('sim-caracter',
                           SIM_OPTIONS['cole_caracter'], 'ACADÉMICO')),
                _sim_field('Bilingüe', _sim_dd('sim-bilingue',
                           SIM_OPTIONS['cole_bilingue'], 'N')),
                _sim_field('Género colegio', _sim_dd('sim-cole-genero',
                           SIM_OPTIONS['cole_genero'], 'MIXTO')),
                _sim_field('Género estudiante', _sim_dd('sim-estu-genero',
                           SIM_OPTIONS['estu_genero'], 'F')),
            ], style={'display': 'flex', 'flexWrap': 'wrap', 'gap': '14px',
                      'marginBottom': '16px'}),

            html.Div([
                _sim_field('Estrato', _sim_dd('sim-estrato',
                           SIM_OPTIONS['fami_estratovivienda'], 'ESTRATO 2')),
                _sim_field('Internet en el hogar', _sim_dd('sim-internet',
                           SIM_OPTIONS['fami_tieneinternet'], 'NO')),
                _sim_field('Computador', _sim_dd('sim-computador',
                           SIM_OPTIONS['fami_tienecomputador'], 'NO')),
                _sim_field('Automóvil', _sim_dd('sim-automovil',
                           SIM_OPTIONS['fami_tieneautomovil'], 'NO')),
                _sim_field('Lavadora', _sim_dd('sim-lavadora',
                           SIM_OPTIONS['fami_tienelavadora'], 'SI')),
            ], style={'display': 'flex', 'flexWrap': 'wrap', 'gap': '14px',
                      'marginBottom': '16px'}),

            html.Div([
                _sim_field('Educación de la madre',
                           _sim_dd('sim-educ-madre', SIM_OPTIONS['fami_educacionmadre'],
                                   'SECUNDARIA (BACHILLERATO) COMPLETA')),
                _sim_field('Educación del padre',
                           _sim_dd('sim-educ-padre', SIM_OPTIONS['fami_educacionpadre'],
                                   'SECUNDARIA (BACHILLERATO) COMPLETA')),
                _sim_field('Cuartos en el hogar',
                           _sim_dd('sim-cuartos', SIM_OPTIONS['fami_cuartoshogar'], 'TRES')),
                _sim_field('Personas en el hogar',
                           _sim_dd('sim-personas', SIM_OPTIONS['fami_personashogar'], '3 A 4')),
                _sim_field('Edad del estudiante',
                           dcc.Input(id='sim-edad', type='number', value=17,
                                     min=14, max=30, debounce=True,
                                     style={'width': '100%', 'padding': '10px 12px',
                                            'border': f'1px solid {BORD}',
                                            'borderRadius': '8px', 'fontSize': '13px'})),
            ], style={'display': 'flex', 'flexWrap': 'wrap', 'gap': '14px',
                      'marginBottom': '22px'}),

            html.Button(
                'Calcular predicción',
                id='sim-predict-btn', n_clicks=0,
                style={
                    'background': NAVY, 'color': 'white', 'border': 'none',
                    'borderRadius': '10px', 'padding': '13px 32px',
                    'fontWeight': '700', 'fontSize': '14px', 'cursor': 'pointer',
                    'letterSpacing': '0.02em', 'boxShadow': '0 2px 6px rgba(30,58,95,0.25)',
                }
            ),

        ], className='chart-card', style={'padding': '24px', 'marginBottom': '20px'}),

        html.Div(id='sim-results'),

    ], className='content-area')


# ─────────────────────────────────────────────────────────────────────
# 6. PRECOMPUTE STATIC CHARTS
# ─────────────────────────────────────────────────────────────────────
FIG_DIST   = _score_dist()
FIG_TREND  = _score_trend()
FIG_ING    = _ingles_levels()
FIG_AREA   = _score_by_area()
FIG_MCPIO  = _mcpio_scores()
FIG_COEFS  = _lasso_coefs()
FIG_CM     = _confusion_matrix()
FIG_IMP    = _feature_importance()
FIG_DONUT  = _mat_donut()
FIG_MDIST  = _mat_dist()
FIG_MAT_CM = _mat_confusion_matrix_model('xgb')
FIG_MAT_METRICS = _mat_metrics_bar('xgb')


# ─────────────────────────────────────────────────────────────────────
# 7. TAB LAYOUTS
# ─────────────────────────────────────────────────────────────────────
def _tab_panorama():
    return html.Div([
        section(
            'Panorama General — Saber 11 Córdoba',
            'Resumen ejecutivo de tres análisis predictivos sobre los resultados del examen '
            'de Estado Saber 11 para estudiantes del departamento de Córdoba. '
            'Los modelos predicen el puntaje global, el nivel de inglés A- y el puntaje '
            'de matemáticas a partir de variables de contexto familiar y escolar.',
        ),

        html.Div([
            kpi('Total registros',   '194.259', 'Estudiantes en la base de datos', NAVY),
            kpi('Municipios',        '33',      'Del departamento de Córdoba', BLUE),
            kpi('Colegios únicos',   '483',     'Sedes ICFES registradas', MID),
            kpi('Períodos cubiertos','2011–2022','11 años de aplicaciones Saber 11', LGRAY),
        ], className='metrics-row'),

        html.Div([
            chart_card(
                'Distribución del puntaje global',
                '118,523 estudiantes con punt_global disponible · media = 241 · mediana = 235 · DE = 46',
                FIG_DIST,
            ),
            chart_card(
                'Tendencia del puntaje promedio por año',
                'Promedio departamental por año de aplicación · banda sombreada = ±1 desviación estándar',
                FIG_TREND,
            ),
        ], className='charts-grid charts-grid-2'),

        html.Div([
            chart_card(
                'Distribución de niveles de inglés',
                'El 61% de los estudiantes se ubica en A- (nivel más bajo del Marco Europeo)',
                FIG_ING, height=300,
            ),
            chart_card(
                'Puntaje promedio por tipo de colegio y área',
                'Brecha privado-oficial ≈ +38 pts · Brecha urbano-rural ≈ +20 pts',
                FIG_AREA, height=300,
            ),
        ], className='charts-grid charts-grid-2'),

        html.Div(className='divider'),

        html.H3('Síntesis por pregunta de investigación',
                style={'fontSize': '1rem', 'fontWeight': '600', 'color': TEXT,
                       'marginBottom': '14px'}),

        html.Div([
            html.Div([
                html.Div('Regresión', className='summary-card-title',
                         style={'color': BLUE}),
                html.Div('Predicción del Puntaje Global Saber 11', className='summary-card-main'),
                html.Div('Red neuronal · selección Lasso · 89 features',
                         className='summary-card-metric'),
                html.Div([
                    html.Span('R² = 0.315', className='summary-card-badge',
                              style={'background': '#dbeafe', 'color': NAVY}),
                    html.Span('RMSE = 38.1 pts', className='summary-card-badge',
                              style={'background': '#eff6ff', 'color': BLUE,
                                     'marginLeft': '7px'}),
                    html.Span('MAPE = 12.9%', className='summary-card-badge',
                              style={'background': '#eff6ff', 'color': BLUE,
                                     'marginLeft': '7px'}),
                ]),
            ], className='summary-card', style={'borderColor': BLUE, 'flex': '1'}),

            html.Div([
                html.Div('Clasificación', className='summary-card-title',
                         style={'color': GRN}),
                html.Div('Predicción del Nivel de Inglés A-', className='summary-card-main'),
                html.Div('MLP 128-64-32 · municipio + encoding de colegio · 137 features',
                         className='summary-card-metric'),
                html.Div([
                    html.Span('F1 = 0.794', className='summary-card-badge',
                              style={'background': '#d1fae5', 'color': GRN}),
                    html.Span('AUC = 0.740', className='summary-card-badge',
                              style={'background': '#f0fdf4', 'color': GRN,
                                     'marginLeft': '7px'}),
                    html.Span('Recall A- = 94.8%', className='summary-card-badge',
                              style={'background': '#f0fdf4', 'color': GRN,
                                     'marginLeft': '7px'}),
                ]),
            ], className='summary-card', style={'borderColor': GRN, 'flex': '1'}),

            html.Div([
                html.Div('Clasificación · en desarrollo', className='summary-card-title',
                         style={'color': AMB}),
                html.Div('Predicción: Puntaje Matemáticas > 70', className='summary-card-main'),
                html.Div('100,094 registros · 2015–2022 · 23 features',
                         className='summary-card-metric'),
                html.Div([
                    html.Span('Desbalanceo 96.6% / 3.4%', className='summary-card-badge',
                              style={'background': '#fef3c7', 'color': AMB}),
                    html.Span('Modelo en proceso', className='summary-card-badge',
                              style={'background': '#fef3c7', 'color': AMB,
                                     'marginLeft': '7px'}),
                ]),
            ], className='summary-card', style={'borderColor': '#d97706', 'flex': '1'}),
        ], style={'display': 'flex', 'gap': '16px', 'marginBottom': '20px'}),

        chart_card(
            'Puntaje promedio por municipio — top 25',
            'Ranking basado en el puntaje global promedio de cada municipio del departamento',
            FIG_MCPIO, height=530,
        ),

    ], className='content-area')


def _tab_regresion():
    return html.Div([
        section(
            'Predicción del Puntaje Global — Regresión',
            'Modelo de redes neuronales para predecir el puntaje global Saber 11 a partir de '
            'variables de contexto familiar y escolar. Se aplicó LassoCV (5-fold) para '
            'selección de variables antes del entrenamiento. División train/val/test: 70/15/15.',
        ),

        html.Div([
            kpi('RMSE  —  test', f'{REG_TEST["rmse"]:.2f} pts',
                'Error cuadrático medio en conjunto de test', RED),
            kpi('MAE  —  test',  f'{REG_TEST["mae"]:.2f} pts',
                'Error absoluto medio en conjunto de test', RED),
            kpi('R²  —  test',   f'{REG_TEST["r2"]:.4f}',
                '31.5% de la varianza explicada por el modelo', BLUE),
            kpi('MAPE  —  test', f'{REG_TEST["mape"]:.1f}%',
                'Error relativo promedio en el conjunto de test', NAVY),
            kpi('Mejor modelo',  'NN-1 (64)',
                'Una capa oculta · 89 features seleccionadas por Lasso', GRN),
        ], className='metrics-row'),

        metric_radio('Métrica de comparación:', 'reg-metric', [
            {'l': 'RMSE', 'v': 'rmse'},
            {'l': 'MAE',  'v': 'mae'},
            {'l': 'R²',   'v': 'r2'},
        ], 'rmse'),

        html.Div([
            html.Div([
                html.Div('Comparación de modelos', className='chart-title'),
                html.Div(
                    'Evaluado en conjunto de validación independiente · '
                    'rojo = mejor modelo · línea punteada = baseline Lasso',
                    className='chart-sub'),
                dcc.Graph(id='reg-chart',
                          figure=_reg_comparison('rmse'),
                          config=_CFG, style={'height': '340px'}),
            ], className='chart-card'),

            chart_card(
                'Coeficientes Lasso — efecto sobre el puntaje',
                'Variables estandarizadas · azul = sube el puntaje · rojo = baja el puntaje · '
                'alpha óptimo = 0.001791 (LassoCV 5-fold)',
                FIG_COEFS, height=340,
            ),
        ], className='charts-grid charts-grid-2'),

        insight(html.Span([
            html.B('Hallazgos del modelo: '),
            'La jornada sabatina (−43 pts) y el calendario A vs. B (−45 pts) son los predictores '
            'negativos más fuertes — en Córdoba los pocos colegios calendario B son instituciones '
            'de perfil alto. Los colegios privados (+21 pts), el posgrado del padre (+27 pts) y '
            'la jornada completa (+16 pts) elevan el puntaje. El R² ≈ 0.31 refleja que el ',
            html.B('69% restante de la varianza es individual e inobservable'),
            ': preparación personal, calidad específica del docente, '
            'condición del estudiante el día del examen. Este techo es esperado y conocido '
            'en la literatura de minería de datos educacionales.',
        ])),

        html.Div(className='divider'),

        html.H3('Proceso de selección de variables — LassoCV',
                style={'fontSize': '0.95rem', 'fontWeight': '600',
                       'color': TEXT, 'marginBottom': '14px'}),

        html.Div([
            kpi('Features tras OHE',    '103',  'Después de codificación categórica', LGRAY),
            kpi('Features seleccionadas','89',  'Con coeficiente Lasso ≠ 0', BLUE),
            kpi('Features eliminadas',   '14',  '13.6% del espacio descartado', MID),
            kpi('Alpha óptimo',       '0.001791', 'LassoCV 5-fold · penalización L1 leve', NAVY),
            kpi('Mejora en RMSE',     '−0.58 pts', 'NN con Lasso vs. NN sin Lasso', GRN),
        ], className='metrics-row'),

    ], className='content-area')


def _tab_ingles():
    return html.Div([
        section(
            'Predicción del Nivel de Inglés A- — Clasificación Binaria',
            'Modelo para identificar, antes de la prueba, qué estudiantes de Córdoba '
            'quedarán en el nivel A- de inglés (el más bajo del Marco Común Europeo). '
            'Uso operativo: focalización de refuerzo pedagógico por la Secretaría de Educación. '
            'División estratificada: train 135,926 / val 29,033 / test 29,111.',
        ),

        html.Div([
            kpi('F1-Score  —  test', f'{ING_TEST["f1"]:.3f}',
                'Métrica principal · clase de riesgo A-', GRN),
            kpi('ROC-AUC  —  test',  f'{ING_TEST["auc"]:.3f}',
                'Capacidad discriminativa del modelo', GRN),
            kpi('Recall A-  —  test', f'{ING_TEST["recall"]:.1%}',
                'De cada 100 en riesgo real, detecta 95', BLUE),
            kpi('Precisión A-',       f'{ING_TEST["precision"]:.1%}',
                'De los flagueados, 68% son riesgo real', MID),
            kpi('Umbral óptimo',      '0.33',
                'Calibrado para maximizar F1 en validación', NAVY),
        ], className='metrics-row'),

        html.Div([
            chart_card(
                'Matriz de confusión — modelo final (test)',
                '29,111 estudiantes · umbral = 0.33 · recall A- = 94.8% · especificidad = 31.2%',
                FIG_CM, height=340,
            ),
            chart_card(
                'Importancia de variables — proxy Random Forest',
                'Estimada sobre 137 features · el encoding histórico del colegio supera en '
                '5× a cualquier variable socioeconómica individual',
                FIG_IMP, height=340,
            ),
        ], className='charts-grid charts-grid-2'),

        metric_radio('Métrica de comparación:', 'ing-metric', [
            {'l': 'F1-Score', 'v': 'f1'},
            {'l': 'ROC-AUC',  'v': 'auc'},
            {'l': 'PR-AUC',   'v': 'pr'},
        ], 'f1'),

        html.Div([
            html.Div([
                html.Div('Comparación de arquitecturas de red neuronal', className='chart-title'),
                html.Div(
                    '8 configuraciones evaluadas en validación independiente · '
                    'rojo = mejor · línea punteada = baseline logístico',
                    className='chart-sub'),
                dcc.Graph(id='ing-chart',
                          figure=_ing_comparison('f1'),
                          config=_CFG, style={'height': '340px'}),
            ], className='chart-card'),
        ], className='charts-grid'),

        insight(html.Span([
            html.B('Hallazgo principal: '),
            'El predictor más importante es la ',
            html.B('tasa histórica de A- del colegio'),
            ' — construida con target encoding bayesiano sobre el set de entrenamiento. '
            'Esto indica que las diferencias entre colegios son ',
            html.B('estructurales y persistentes'),
            ', no explicadas únicamente por la composición socioeconómica. '
            'Con recall del 94.8%, el modelo detecta a ',
            html.B('95 de cada 100 estudiantes en riesgo real'),
            ', habilitando intervención pedagógica focalizada antes de la prueba. '
            'El internet en el hogar y el computador superan al estrato directo como '
            'predictores: la conectividad doméstica facilita el consumo de contenido '
            'en inglés fuera del aula.',
        ])),

        html.Div(className='divider'),

        html.H3('Variables y evolución del modelo',
                style={'fontSize': '0.95rem', 'fontWeight': '600',
                       'color': TEXT, 'marginBottom': '14px'}),

        html.Div([
            kpi('Features v1',       '102', 'OHE · contexto familiar y escolar', LGRAY),
            kpi('Features v2',       '137', '+municipio OHE + encoding colegio', BLUE),
            kpi('Ganancia AUC v1→v2', '+0.026', '0.7074 → 0.7350 al agregar features locales', GRN),
            kpi('Ganancia F1 v1→v2',  '+0.039', '0.7540 → 0.7926 con threshold óptimo', GRN),
            kpi('Datos de entrenamiento', '165K', 'train + val para el modelo final', NAVY),
        ], className='metrics-row'),

        chart_card(
            'Distribución de niveles de inglés en Córdoba',
            '61% de los estudiantes en el nivel más bajo A- · solo el 2% alcanza B o superior',
            FIG_ING, height=300,
        ),

    ], className='content-area')


def _tab_matematicas():
    return html.Div([
        section(
            'Predicción del Puntaje de Matemáticas > 70 — Clasificación Binaria',
            'Modelo para identificar si un estudiante de Córdoba alcanza un puntaje superior '
            'a 70 en matemáticas Saber 11. El objetivo operativo es detectar estudiantes con '
            'preparación suficiente para avanzar a cálculo diferencial, priorizando el recall '
            'para reducir falsos negativos.'
        ),

        metric_radio('Modelo a evaluar:', 'mat-model-selector', [
            {'l': 'Modelo XGBoost', 'v': 'xgb'},
            {'l': 'Modelo de Redes Neuronales', 'v': 'nn'},
        ], 'xgb'),

        html.Div(
            _mat_kpi_cards('xgb'),
            id='mat-kpis',
            className='metrics-row'
        ),

        html.Div([
            html.Div([
                html.Div('Matriz de confusión — modelo seleccionado (test)',
                         className='chart-title'),
                html.Div(
                    'Conjunto de test: 5,943 estudiantes · clase positiva: 235 estudiantes · '
                    'positivo = puntaje de matemáticas > 70',
                    className='chart-sub'
                ),
                dcc.Graph(
                    id='mat-cm-chart',
                    figure=FIG_MAT_CM,
                    config=_CFG,
                    style={'height': '340px'}
                ),
            ], className='chart-card'),

            html.Div([
                html.Div('Métricas del modelo seleccionado',
                         className='chart-title'),
                html.Div(
                    'La métrica principal es recall, porque un falso negativo asignaría '
                    'refuerzo a un estudiante que sí cumple el requisito.',
                    className='chart-sub'
                ),
                dcc.Graph(
                    id='mat-metrics-chart',
                    figure=FIG_MAT_METRICS,
                    config=_CFG,
                    style={'height': '340px'}
                ),
            ], className='chart-card'),
        ], className='charts-grid charts-grid-2'),

        html.Div(
            id='mat-model-note',
            children=insight(html.Span([
                html.B('Modelo seleccionado: '),
                'XGBoost con umbral optimizado. En test alcanza recall de 82.6% y AUC de 0.886, '
                'superando a la red neuronal en detección de estudiantes que sí cumplen el umbral.'
            ]), kind='success')
        ),

        html.Div(className='divider'),

        html.Div([
            chart_card(
                'Balance de clases — variable objetivo',
                'Desbalanceo severo: solo el 3.95% supera el umbral de 70 pts en matemáticas',
                FIG_DONUT,
                height=320,
            ),
            chart_card(
                'Distribución del puntaje de matemáticas',
                'Rojo = no cumple el umbral (≤ 70) · Verde = cumple (> 70) · línea punteada = umbral',
                FIG_MDIST,
                height=320,
            ),
        ], className='charts-grid charts-grid-2'),

        insight(html.Span([
            html.B('Interpretación del modelo: '),
            'La exactitud no es la métrica principal porque el problema está fuertemente '
            'desbalanceado. Un clasificador que prediga siempre "no cumple" tendría alta '
            'exactitud, pero sería inútil. Por eso se priorizan recall, F1-score y AUC.'
        ]), kind='warn'),

        html.Div(className='divider'),

        html.Div([
            html.Div('Simulador de predicción individual',
                     className='chart-title'),
            html.Div(
                'Ingrese las características del estudiante. El sistema transforma las variables '
                'a la misma codificación usada en entrenamiento y estima si el estudiante cumple '
                'el umbral de matemáticas.',
                className='chart-sub',
                style={'marginBottom': '10px'}
            ),

            *[
                _mat_input_section(title, cols)
                for title, cols in MAT_FEATURE_GROUPS
            ],

            html.Button(
                'Predecir resultado',
                id='mat-predict-button',
                n_clicks=0,
                style={
                    'marginTop': '18px',
                    'background': NAVY,
                    'color': 'white',
                    'border': 'none',
                    'borderRadius': '10px',
                    'padding': '12px 18px',
                    'fontWeight': '600',
                    'fontSize': '14px',
                    'cursor': 'pointer'
                }
            ),

            html.Div(
                id='mat-prediction-output',
                style={'marginTop': '18px'}
            )

        ], className='chart-card', style={'padding': '22px'}),

    ], className='content-area')


# ─────────────────────────────────────────────────────────────────────
# 8. APP LAYOUT
# ─────────────────────────────────────────────────────────────────────
app = dash.Dash(
    __name__,
    title='Saber 11 — Córdoba',
    suppress_callback_exceptions=True,
)

app.layout = html.Div([
    html.Div([
        html.Div([
            html.Div('Saber 11 — Córdoba', className='app-header-title'),
            html.Div('Análisis predictivo · Secretaría de Educación',
                     className='app-header-sub'),
        ], className='app-header-left'),
        html.Div([
            html.Div('194.259 estudiantes', className='header-badge'),
            html.Div('2011 – 2022',         className='header-badge'),
            html.Div('Departamento de Córdoba', className='header-badge'),
        ], className='app-header-right'),
    ], className='app-header'),

    html.Div([
        dcc.Tabs(
            id='tabs',
            value='panorama',
            className='custom-tabs',
            children=[
                dcc.Tab(label='Panorama General',    value='panorama'),
                dcc.Tab(label='Puntaje Global',      value='regresion'),
                dcc.Tab(label='Nivel de Inglés A-',  value='ingles'),
                dcc.Tab(label='Matemáticas',         value='matematicas'),
                dcc.Tab(label='Simulador',           value='simulador'),
            ],
        ),
    ], className='tab-container'),

    dcc.Loading(
        html.Div(id='tab-content'),
        type='dot',
        color=BLUE,
    ),
], style={'background': '#f7f8fb', 'minHeight': '100vh'})


# ─────────────────────────────────────────────────────────────────────
# 9. CALLBACKS
# ─────────────────────────────────────────────────────────────────────
@app.callback(Output('tab-content', 'children'), Input('tabs', 'value'))
def render_tab(tab):
    if tab == 'regresion':   return _tab_regresion()
    if tab == 'ingles':      return _tab_ingles()
    if tab == 'matematicas': return _tab_matematicas()
    if tab == 'simulador':   return _tab_simulador()
    return _tab_panorama()


@app.callback(Output('reg-chart', 'figure'), Input('reg-metric', 'value'))
def update_reg(metric):
    return _reg_comparison(metric or 'rmse')


@app.callback(Output('ing-chart', 'figure'), Input('ing-metric', 'value'))
def update_ing(metric):
    return _ing_comparison(metric or 'f1')

@app.callback(
    Output('mat-kpis', 'children'),
    Output('mat-cm-chart', 'figure'),
    Output('mat-metrics-chart', 'figure'),
    Output('mat-model-note', 'children'),
    Input('mat-model-selector', 'value')
)
def update_mat_model(model_key):
    m = _mat_model_data(model_key)

    note_kind = 'success' if model_key == 'xgb' else 'info'

    note = insight(html.Span([
        html.B(f'{m["label"]}: '),
        m['note']
    ]), kind=note_kind)

    return (
        _mat_kpi_cards(model_key),
        _mat_confusion_matrix_model(model_key),
        _mat_metrics_bar(model_key),
        note
    )


@app.callback(
    Output('mat-prediction-output', 'children'),
    Input('mat-predict-button', 'n_clicks'),
    State('mat-model-selector', 'value'),
    *[State(f'mat-input-{col}', 'value') for col in MAT_FEATURES],
    prevent_initial_call=True
)
def predict_math_score(n_clicks, model_key, *values):
    if not n_clicks:
        raise PreventUpdate

    if not COLUMNAS_MODELO:
        return insight(html.Span([
            html.B('Error: '),
            'No se encontró columnas_modelo.json. Este archivo es necesario para alinear '
            'las variables del formulario con las columnas usadas en entrenamiento.'
        ]), kind='warn')

    raw_data = dict(zip(MAT_FEATURES, values))

    for col in MAT_NUMERIC_FEATURES:
        raw_data[col] = pd.to_numeric(raw_data.get(col), errors='coerce')

    nuevo_estudiante = pd.DataFrame([raw_data])

    nuevo_estudiante_encoded = pd.get_dummies(nuevo_estudiante)

    nuevo_estudiante_encoded = nuevo_estudiante_encoded.reindex(
        columns=COLUMNAS_MODELO,
        fill_value=0
    )

    nuevo_estudiante_encoded = nuevo_estudiante_encoded.astype('float32')

    if model_key == 'xgb':
        if MAT_XGB_MODEL is None:
            return insight(html.Span([
                html.B('Modelo no disponible: '),
                'No se pudo cargar modelo_final_icfes_xgboost.joblib.'
            ]), kind='warn')

        probabilidad = float(
            MAT_XGB_MODEL.predict_proba(nuevo_estudiante_encoded)[:, 1][0]
        )
        threshold = MAT_XGB_THRESHOLD
        model_label = 'Modelo XGBoost'

    else:
        if MAT_NN_MODEL is None:
            return insight(html.Span([
                html.B('Modelo no disponible: '),
                'No se pudo cargar modelo_final_icfes.keras. Revise que TensorFlow esté instalado.'
            ]), kind='warn')

        probabilidad = float(
            MAT_NN_MODEL.predict(nuevo_estudiante_encoded, verbose=0).ravel()[0]
        )
        threshold = MAT_NN_THRESHOLD
        model_label = 'Modelo de Redes Neuronales'

    prediccion = 1 if probabilidad >= threshold else 0

    if prediccion == 1:
        titulo = 'Predicción: Cumple el requisito'
        mensaje = (
            'El modelo estima que el estudiante tiene probabilidad suficiente de superar '
            'el umbral de matemáticas (> 70).'
        )
        color = GRN
        kind = 'success'
    else:
        titulo = 'Predicción: Requiere refuerzo'
        mensaje = (
            'El modelo estima que el estudiante no supera el umbral de matemáticas (> 70). '
            'Se recomienda considerar refuerzo o precálculo.'
        )
        color = RED
        kind = 'warn'

    return html.Div([
        html.Div(titulo, className='summary-card-main',
                 style={'color': color, 'marginBottom': '8px'}),
        html.Div([
            html.Span(f'Modelo usado: {model_label}',
                      className='summary-card-badge',
                      style={'background': '#eff6ff', 'color': NAVY}),
            html.Span(f'Probabilidad estimada: {probabilidad:.1%}',
                      className='summary-card-badge',
                      style={'background': '#f0fdf4', 'color': GRN,
                             'marginLeft': '7px'}),
            html.Span(f'Umbral: {threshold:.3f}',
                      className='summary-card-badge',
                      style={'background': '#fef3c7', 'color': AMB,
                             'marginLeft': '7px'}),
        ], style={'marginBottom': '10px'}),
        insight(html.Span(mensaje), kind=kind)
    ], className='summary-card', style={'borderColor': color})

# ── Simulator: update school options when municipality changes ────────
@app.callback(
    Output('sim-school', 'options'),
    Output('sim-school', 'value'),
    Input('sim-mcpio', 'value'),
)
def update_sim_schools(mcpio):
    schools = school_by_mcpio.get(mcpio, []) if mcpio else []
    opts    = [{'label': s, 'value': s} for s in schools]
    val     = schools[0] if schools else None
    return opts, val


# ── Simulator: run both models on button click ────────────────────────
@app.callback(
    Output('sim-results', 'children'),
    Input('sim-predict-btn', 'n_clicks'),
    State('sim-school',      'value'),
    State('sim-mcpio',       'value'),
    State('sim-area',        'value'),
    State('sim-naturaleza',  'value'),
    State('sim-jornada',     'value'),
    State('sim-calendario',  'value'),
    State('sim-bilingue',    'value'),
    State('sim-caracter',    'value'),
    State('sim-cole-genero', 'value'),
    State('sim-estu-genero', 'value'),
    State('sim-estrato',     'value'),
    State('sim-internet',    'value'),
    State('sim-computador',  'value'),
    State('sim-automovil',   'value'),
    State('sim-lavadora',    'value'),
    State('sim-educ-madre',  'value'),
    State('sim-educ-padre',  'value'),
    State('sim-cuartos',     'value'),
    State('sim-personas',    'value'),
    State('sim-edad',        'value'),
    prevent_initial_call=True,
)
def predict_simulador(n_clicks, school, mcpio, area, naturaleza, jornada,
                      calendario, bilingue, caracter, cole_genero, estu_genero,
                      estrato, internet, computador, automovil, lavadora,
                      educ_madre, educ_padre, cuartos, personas, edad):
    if not n_clicks:
        raise PreventUpdate

    if ING_V2_MODEL is None or REG_V2_MODEL is None:
        return insight(html.Span([
            html.B('Modelos no disponibles. '),
            'Verifique que TensorFlow esté instalado y los archivos .keras estén en '
            'Tarea4_Modelos/modelos/.',
        ]), kind='warn')

    try:
        prob_ing, pred_global = _predict_both(
            school or '', mcpio or '', area, naturaleza, jornada, calendario,
            bilingue, caracter, cole_genero, estu_genero, estrato,
            educ_madre, educ_padre, internet, computador, automovil,
            lavadora, cuartos, personas, edad or 17,
        )
    except Exception as exc:
        return insight(html.Span([html.B('Error de predicción: '), str(exc)]), kind='warn')

    # ── Math model (XGBoost) ──────────────────────────────────────────
    math_prob = math_label = None
    if MAT_XGB_MODEL is not None and COLUMNAS_MODELO:
        try:
            raw_math = {
                'anio_periodo':       2022,
                'periodo_aplicacion': 1,
                'cole_area_ubicacion': area,
                'cole_bilingue':      bilingue,
                'cole_calendario':    calendario,
                'cole_caracter':      caracter,
                'cole_genero':        cole_genero,
                'cole_jornada':       jornada,
                'cole_naturaleza':    naturaleza,
                'cole_sede_principal': 'S',
                'cole_mcpio_ubicacion': mcpio,
                'edad_aprox':         float(edad or 17),
                'estu_genero':        estu_genero,
                'estu_mcpio_reside':  mcpio,
                'fami_cuartoshogar':  cuartos,
                'fami_educacionmadre': educ_madre,
                'fami_educacionpadre': educ_padre,
                'fami_estratovivienda': estrato,
                'fami_personashogar': personas,
                'fami_tieneautomovil': automovil,
                'fami_tienecomputador': computador,
                'fami_tieneinternet': internet,
                'fami_tienelavadora': lavadora,
            }
            _dfm = pd.get_dummies(pd.DataFrame([raw_math]))
            _dfm = _dfm.reindex(columns=COLUMNAS_MODELO, fill_value=0).astype('float32')
            math_prob  = float(MAT_XGB_MODEL.predict_proba(_dfm)[:, 1][0])
            math_label = ('Cumple el umbral (> 70)'
                          if math_prob >= MAT_XGB_THRESHOLD
                          else 'No cumple el umbral (≤ 70)')
        except Exception as _me:
            print(f'[!] Math sim error: {_me}')

    threshold_ing = float(ING_V2_META['threshold'])
    is_riesgo     = prob_ing >= threshold_ing
    dept_mean     = 241.0
    dept_rmse     = 36.3

    score_color = GRN if pred_global >= dept_mean else RED
    ing_color   = RED if is_riesgo else GRN
    ing_bg      = '#fef2f2' if is_riesgo else '#f0fdf4'
    ing_label   = 'Alto riesgo de nivel A-' if is_riesgo else 'Bajo riesgo de nivel A-'
    diff        = pred_global - dept_mean
    diff_str    = f'+{diff:.0f}' if diff >= 0 else f'{diff:.0f}'

    gauge_ing = go.Figure(go.Indicator(
        mode='gauge+number',
        value=round(prob_ing * 100, 1),
        number={'suffix': '%', 'font': {'size': 28, 'color': ing_color}},
        gauge=dict(
            axis=dict(range=[0, 100], tickwidth=1, tickcolor='#ccc',
                      tickfont=dict(size=10)),
            bar=dict(color=ing_color, thickness=0.3),
            bgcolor='white', borderwidth=0,
            threshold=dict(line=dict(color=NAVY, width=3), thickness=0.75,
                           value=round(threshold_ing * 100, 1)),
            steps=[
                {'range': [0,  threshold_ing * 100], 'color': '#f0fdf4'},
                {'range': [threshold_ing * 100, 100], 'color': '#fef2f2'},
            ],
        ),
    ))
    gauge_ing.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=_L['font'], margin=dict(l=20, r=20, t=20, b=20), height=200,
    )

    math_card = html.Div([], style={'display': 'none'})
    if math_prob is not None:
        meets = math_prob >= MAT_XGB_THRESHOLD
        mat_color = GRN if meets else RED
        mat_bg    = '#f0fdf4' if meets else '#fef2f2'
        gauge_mat = go.Figure(go.Indicator(
            mode='gauge+number',
            value=round(math_prob * 100, 1),
            number={'suffix': '%', 'font': {'size': 28, 'color': mat_color}},
            gauge=dict(
                axis=dict(range=[0, 100], tickwidth=1, tickcolor='#ccc',
                          tickfont=dict(size=10)),
                bar=dict(color=mat_color, thickness=0.3),
                bgcolor='white', borderwidth=0,
                threshold=dict(line=dict(color=NAVY, width=3), thickness=0.75,
                               value=round(MAT_XGB_THRESHOLD * 100, 1)),
                steps=[
                    {'range': [0,  MAT_XGB_THRESHOLD * 100], 'color': '#fef2f2'},
                    {'range': [MAT_XGB_THRESHOLD * 100, 100], 'color': '#f0fdf4'},
                ],
            ),
        ))
        gauge_mat.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=_L['font'], margin=dict(l=20, r=20, t=20, b=20), height=200,
        )
        math_card = html.Div([
            html.Div('Puntaje Matemáticas > 70', className='metric-label'),
            dcc.Graph(figure=gauge_mat, config={'displayModeBar': False},
                      style={'height': '190px', 'marginTop': '4px',
                             'marginBottom': '4px'}),
            html.Div([
                html.Span(math_label, className='summary-card-badge',
                          style={'background': mat_bg, 'color': mat_color,
                                 'fontSize': '12px', 'fontWeight': '700'}),
                html.Span(f'Umbral = {MAT_XGB_THRESHOLD:.2f}',
                          className='summary-card-badge',
                          style={'background': '#f1f5f9', 'color': NAVY,
                                 'marginLeft': '8px', 'fontSize': '11px'}),
            ]),
        ], className='metric-card',
           style={'borderTop': f'3px solid {mat_color}', 'flex': '1',
                  'padding': '22px', 'background': mat_bg})

    math_insight = []
    if math_prob is not None:
        meets = math_prob >= MAT_XGB_THRESHOLD
        math_insight = [
            ' El modelo XGBoost estima una probabilidad de ',
            html.B(f'{math_prob:.1%}'),
            ' de que el estudiante supere el umbral de 70 pts en matemáticas — ',
            html.B('cumple el requisito' if meets else 'no cumple el requisito'),
            f' (umbral = {MAT_XGB_THRESHOLD:.2f}).',
        ]

    return html.Div([
        html.Div([
            html.Div([
                html.Div('Puntaje Global Estimado', className='metric-label'),
                html.Div(f'{pred_global:.0f}',
                         style={'fontSize': '3rem', 'fontWeight': '800',
                                'color': score_color, 'lineHeight': '1',
                                'letterSpacing': '-0.03em', 'marginBottom': '8px'}),
                html.Div('puntos', style={'fontSize': '13px', 'color': SUB,
                                         'marginTop': '-4px', 'marginBottom': '12px'}),
                html.Div([
                    html.Span('Media departamental: 241 pts',
                              style={'fontSize': '12px', 'color': SUB}),
                    html.Span(f' {diff_str} pts',
                              style={'fontSize': '13px', 'fontWeight': '700',
                                     'color': score_color, 'marginLeft': '8px'}),
                ]),
                html.Div(
                    f'Intervalo: [{pred_global - dept_rmse:.0f} – {pred_global + dept_rmse:.0f}] pts  (±1 RMSE)',
                    style={'fontSize': '11px', 'color': LGRAY, 'marginTop': '8px'}
                ),
            ], className='metric-card',
               style={'borderTop': f'3px solid {score_color}', 'flex': '1',
                      'padding': '22px'}),

            html.Div([
                html.Div('Probabilidad de Nivel A- en Inglés', className='metric-label'),
                dcc.Graph(figure=gauge_ing, config={'displayModeBar': False},
                          style={'height': '190px', 'marginTop': '4px',
                                 'marginBottom': '4px'}),
                html.Div([
                    html.Span(ing_label, className='summary-card-badge',
                              style={'background': ing_bg, 'color': ing_color,
                                     'fontSize': '12px', 'fontWeight': '700'}),
                    html.Span(f'Umbral = {threshold_ing:.2f}',
                              className='summary-card-badge',
                              style={'background': '#f1f5f9', 'color': NAVY,
                                     'marginLeft': '8px', 'fontSize': '11px'}),
                ]),
            ], className='metric-card',
               style={'borderTop': f'3px solid {ing_color}', 'flex': '1',
                      'padding': '22px', 'background': ing_bg}),

            math_card,

        ], className='metrics-row'),

        insight(html.Span([
            html.B('Interpretación: '),
            'El modelo de regresión estima un puntaje global de ',
            html.B(f'{pred_global:.0f} pts'),
            f' ({diff_str} pts respecto a la media departamental de 241). '
            'El modelo de clasificación estima una probabilidad de ',
            html.B(f'{prob_ing:.1%}'),
            ' de que el estudiante quede en nivel A- en inglés — ',
            html.B(ing_label.lower()),
            f' (umbral = {threshold_ing:.2f}).',
            *math_insight,
        ])),
    ])


# ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=8050)
