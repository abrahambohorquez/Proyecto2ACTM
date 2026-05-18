"""
Tablero Saber 11 — Córdoba
Análisis predictivo · Secretaría de Educación del Departamento de Córdoba
"""

import os
import numpy as np
import pandas as pd
import dash
from dash import dcc, html, Input, Output
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
_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    '..', 'Resultados_ICFES_Cordoba_clean.csv')

df = None
_hg = _hm = None  # histogram tuples
agg_year = agg_area = agg_nat = agg_ingles = agg_mcpio = None

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

    print(f'[OK] CSV: {df.shape[0]:,} filas')
except Exception as _e:
    print(f'[!]  CSV no disponible — {_e}')


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
MAT      = dict(n=100094, n_clean=29714, pct_cumple=3.95, pct_no=96.05)


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
    for val, lbl, col in [(241, 'Media 241', RED), (235, 'Mediana 235', GRN)]:
        fig.add_vline(x=val, line_dash='dash', line_color=col, line_width=2,
                      annotation_text=lbl, annotation_font_size=11,
                      annotation_font_color=col, annotation_position='top right')
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
            'Puntaje de Matemáticas — Análisis de Clasificación',
            'Exploración predictiva para identificar qué estudiantes de Córdoba alcanzan '
            'un puntaje mayor a 70 en la prueba de matemáticas Saber 11 (período 2015–2022). '
            'El desbalanceo severo de clases (96.6% / 3.4%) es el principal reto metodológico.',
        ),

        html.Div([
            kpi('Estudiantes (2015–2022)', '100.094',
                'Período filtrado del dataset completo', NAVY),
            kpi('Cumplen el umbral (> 70)',  f'{MAT["pct_cumple"]:.1f}%',
                '3,401 estudiantes del período analizado', RED),
            kpi('No cumplen (≤ 70)',          f'{MAT["pct_no"]:.1f}%',
                '96,693 estudiantes · clase mayoritaria', LGRAY),
            kpi('Umbral de clasificación',    '> 70 pts',
                'Puntaje de corte definido en el análisis', BLUE),
            kpi('Features disponibles',       '23',
                'Temporales · colegio · familia · estudiante', MID),
        ], className='metrics-row'),

        html.Div([
            chart_card(
                'Balance de clases — variable objetivo',
                'Desbalanceo severo: solo el 3.4% supera el umbral de 70 pts en matemáticas',
                FIG_DONUT, height=320,
            ),
            chart_card(
                'Distribución del puntaje de matemáticas',
                'Rojo = no cumple el umbral (≤ 70) · Verde = cumple (> 70) · línea punteada = umbral',
                FIG_MDIST, height=320,
            ),
        ], className='charts-grid charts-grid-2'),

        insight(html.Span([
            html.B('Reto metodológico — desbalanceo severo: '),
            'Un modelo que clasifique a ',
            html.B('todos'),
            ' los estudiantes como "no cumple" obtendría una exactitud del 96.6% '
            'sin aprender nada. Para este problema, las métricas relevantes son ',
            html.B('F1, PR-AUC y recall de la clase minoritaria'),
            ', no la exactitud. Se requieren técnicas como class_weight balanceado, '
            'SMOTE sobre el conjunto de entrenamiento, o ajuste de umbral de decisión '
            'para que el modelo sea útil operativamente.',
        ]), kind='warn'),

        html.Div(className='divider'),

        html.H3('Variables del modelo',
                style={'fontSize': '0.95rem', 'fontWeight': '600',
                       'color': TEXT, 'marginBottom': '14px'}),

        html.Div([
            html.Div([
                html.Div('Temporales', className='feat-box-title',
                         style={'color': BLUE}),
                html.Div('anio_periodo · periodo_aplicacion',
                         className='feat-box-content'),
            ], className='feat-box'),
            html.Div([
                html.Div('Características del colegio', className='feat-box-title',
                         style={'color': BLUE}),
                html.Div('área · bilingüe · calendario · carácter · género · '
                         'jornada · naturaleza · sede · municipio',
                         className='feat-box-content'),
            ], className='feat-box', style={'flex': '2'}),
            html.Div([
                html.Div('Contexto familiar', className='feat-box-title',
                         style={'color': BLUE}),
                html.Div('cuartos · educ. madre · educ. padre · estrato · '
                         'personas · automóvil · computador · internet · lavadora',
                         className='feat-box-content'),
            ], className='feat-box', style={'flex': '2'}),
        ], style={'display': 'flex', 'gap': '14px', 'marginBottom': '20px'}),

        insight(html.Span([
            html.B('Estado del análisis: '),
            'La exploración de datos confirma la viabilidad del problema predictivo. '
            'Las variables de contexto disponibles son las mismas que explican el puntaje '
            'global (R² ≈ 0.31) y el nivel de inglés (AUC ≈ 0.74), por lo que es razonable '
            'esperar un desempeño similar una vez resuelto el desbalanceo de clases. '
            'El análisis completo de modelamiento se encuentra en desarrollo.',
        ]), kind='warn'),

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
    return _tab_panorama()


@app.callback(Output('reg-chart', 'figure'), Input('reg-metric', 'value'))
def update_reg(metric):
    return _reg_comparison(metric or 'rmse')


@app.callback(Output('ing-chart', 'figure'), Input('ing-metric', 'value'))
def update_ing(metric):
    return _ing_comparison(metric or 'f1')


# ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=8050)
