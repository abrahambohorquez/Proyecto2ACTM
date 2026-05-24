"""
Saber 11 Córdoba — Simulador para Estudiantes
Conoce tu predicción y descubre cómo mejorar tu puntaje antes del examen.
"""

import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import joblib

st.set_page_config(
    page_title='Simulador Saber 11 — Córdoba',
    layout='wide',
    initial_sidebar_state='expanded',
)

st.markdown("""
<style>
[data-testid="stSidebar"] { background: #1a2c42; }
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] div { color: #e2e8f0 !important; }
[data-testid="stSidebar"] .stSelectbox label { color: #94a3b8 !important; font-size: 0.78rem !important; }
h1 { color: #1e3a5f !important; letter-spacing: -0.02em; }
h2, h3 { color: #1e3a5f !important; }
.tip-card  { background:#f0fdf4; border-left:3px solid #2d6a4f;
             border-radius:0 8px 8px 0; padding:13px 16px; margin:8px 0; }
.warn-card { background:#fffbeb; border-left:3px solid #d97706;
             border-radius:0 8px 8px 0; padding:13px 16px; margin:8px 0; }
.tip-title { font-weight:700; color:#111827; font-size:0.92rem; }
.tip-body  { font-size:0.875rem; color:#374151; margin-top:4px; line-height:1.65; }
</style>
""", unsafe_allow_html=True)


# ── Cargar modelos ─────────────────────────────────────────────────────────────
_DIR  = os.path.dirname(os.path.abspath(__file__))
_MDIR = os.path.join(_DIR, 'modelos')

@st.cache_resource(show_spinner='Cargando modelos de predicción…')
def load_models():
    import onnxruntime as ort
    ing_sess = ort.InferenceSession(os.path.join(_MDIR, 'modelo_ingles_A-_v2.onnx'))
    ing_prep  = joblib.load(os.path.join(_MDIR, 'preprocessor_ingles_A-_v2.pkl'))
    ing_meta  = joblib.load(os.path.join(_MDIR, 'metadata_ingles_A-_v2.pkl'))
    reg_sess = ort.InferenceSession(os.path.join(_MDIR, 'modelo_puntaje_global_v2.onnx'))
    reg_prep  = joblib.load(os.path.join(_MDIR, 'preprocessor_puntaje_global_v2.pkl'))
    reg_meta  = joblib.load(os.path.join(_MDIR, 'metadata_puntaje_global_v2.pkl'))
    return ing_sess, ing_prep, ing_meta, reg_sess, reg_prep, reg_meta

try:
    ing_model, ing_prep, ing_meta, reg_model, reg_prep, reg_meta = load_models()
except Exception as exc:
    st.error(f'No se pudieron cargar los modelos: {exc}')
    st.info('Asegúrate de que la carpeta `modelos/` contiene los archivos .onnx y .pkl.')
    st.stop()

DEPT_MEAN  = 241.0
RMSE       = 36.3
ING_THRESH = float(ing_meta['threshold'])

MCPIOS = sorted([
    'AYAPEL','BUENAVISTA','CANALETE','CERETÉ','CHIMÁ','CHINÚ',
    'CIÉNAGA DE ORO','COTORRA','LA APARTADA','LORICA','LOS CÓRDOBAS',
    'MOMIL','MONTELÍBANO','MONTERÍA','MOÑITOS','PLANETA RICA',
    'PUEBLO NUEVO','PUERTO ESCONDIDO','PUERTO LIBERTADOR','PURÍSIMA',
    'SAHAGÚN','SAN ANDRÉS DE SOTAVENTO','SAN ANTERO',
    'SAN BERNARDO DEL VIENTO','SAN CARLOS','SAN JOSÉ DE URÉ',
    'SAN PELAYO','TIERRALTA','TUCHÍN','VALENCIA',
])


# ── Función de predicción ──────────────────────────────────────────────────────
def _onnx_run(session, X) -> float:
    import numpy as np
    inp = session.get_inputs()[0].name
    return float(session.run(None, {inp: X.astype(np.float32)})[0][0][0])

def _predict(profile: dict, school: str) -> tuple:
    enc_ing  = ing_meta['school_encoding'].get(school, ing_meta['p_global'])
    X_ing    = ing_prep.transform(pd.DataFrame([{**profile, 'colegio_enc': enc_ing}]))
    prob_ing = _onnx_run(ing_model, X_ing)

    enc_reg  = reg_meta['school_encoding'].get(school, reg_meta['p_global'])
    X_reg    = reg_prep.transform(pd.DataFrame([{**profile, 'colegio_enc': enc_reg}]))
    score    = _onnx_run(reg_model, X_reg)
    return prob_ing, score


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('## Tu perfil')
    nombre = st.text_input('Nombre (opcional)', placeholder='Ej: Mariana López')

    st.markdown('---')
    st.markdown('**Colegio**')
    mcpio   = st.selectbox('Municipio', MCPIOS, index=MCPIOS.index('MONTERÍA'))
    school  = st.text_input('Nombre del colegio', placeholder='Escribe el nombre exacto',
                            help='Si no coincide exactamente, se usará el promedio del municipio.')
    area    = st.selectbox('Área', ['URBANO', 'RURAL'])
    nat     = st.selectbox('Naturaleza', ['OFICIAL', 'NO OFICIAL'])
    jornada = st.selectbox('Jornada', ['MAÑANA', 'COMPLETA', 'TARDE', 'NOCHE', 'SABATINA', 'UNICA'])
    cal     = st.selectbox('Calendario', ['A', 'B'])
    caracter= st.selectbox('Carácter', ['ACADÉMICO', 'TÉCNICO', 'TÉCNICO/ACADÉMICO', 'NO APLICA'])
    bilingue= st.selectbox('Bilingüe', ['N', 'S'])
    cole_gen= st.selectbox('Género del colegio', ['MIXTO', 'FEMENINO'])

    st.markdown('---')
    st.markdown('**Estudiante**')
    estu_gen= st.selectbox('Género', ['F', 'M'])
    edad    = st.slider('Edad', 14, 25, 17)

    st.markdown('---')
    st.markdown('**Familia y hogar**')
    estrato  = st.selectbox('Estrato', ['ESTRATO 1','ESTRATO 2','ESTRATO 3',
                                         'ESTRATO 4','ESTRATO 5','ESTRATO 6','SIN ESTRATO'])
    internet = st.selectbox('Internet en el hogar', ['NO', 'SI'])
    comp     = st.selectbox('Computador en el hogar', ['NO', 'SI'])
    auto     = st.selectbox('Automóvil', ['NO', 'SI'])
    lavadora = st.selectbox('Lavadora', ['SI', 'NO'])
    cuartos  = st.selectbox('Cuartos', ['UNO','DOS','TRES','CUATRO','CINCO',
                                         'SEIS','SIETE','OCHO','NUEVE','DIEZ O MÁS'])
    personas = st.selectbox('Personas en el hogar', ['1 A 2','3 A 4','5 A 6','7 A 8','9 O MÁS'])
    EDUC_OPT = [
        'NINGUNO','PRIMARIA INCOMPLETA','PRIMARIA COMPLETA',
        'SECUNDARIA (BACHILLERATO) INCOMPLETA','SECUNDARIA (BACHILLERATO) COMPLETA',
        'TÉCNICA O TECNOLÓGICA INCOMPLETA','TÉCNICA O TECNOLÓGICA COMPLETA',
        'EDUCACIÓN PROFESIONAL INCOMPLETA','EDUCACIÓN PROFESIONAL COMPLETA',
        'POSTGRADO','NO APLICA','NO SABE',
    ]
    educ_mad = st.selectbox('Educación de la madre', EDUC_OPT, index=4)
    educ_pad = st.selectbox('Educación del padre',   EDUC_OPT, index=4)

    st.markdown('---')
    calcular = st.button('Calcular mi predicción', type='primary', use_container_width=True)


# ── Perfil base ────────────────────────────────────────────────────────────────
profile = {
    'cole_area_ubicacion':  area,
    'cole_naturaleza':      nat,
    'cole_jornada':         jornada,
    'cole_calendario':      cal,
    'cole_caracter':        caracter,
    'cole_bilingue':        bilingue,
    'cole_genero':          cole_gen,
    'estu_genero':          estu_gen,
    'fami_tieneinternet':   internet,
    'fami_tienecomputador': comp,
    'fami_tieneautomovil':  auto,
    'fami_tienelavadora':   lavadora,
    'cole_mcpio_ubicacion': mcpio,
    'fami_estratovivienda': estrato,
    'fami_educacionmadre':  educ_mad,
    'fami_educacionpadre':  educ_pad,
    'fami_cuartoshogar':    cuartos,
    'fami_personashogar':   personas,
    'edad':                 float(edad),
}
school_key = school.strip().upper() if school.strip() else ''


# ── Header ─────────────────────────────────────────────────────────────────────
greeting = f', {nombre.strip()}' if nombre.strip() else ''
st.title(f'Simulador Saber 11 — Córdoba')
if nombre.strip():
    st.markdown(f'### Hola{greeting}, este es tu perfil predictivo para el examen.')
st.caption(
    'Los modelos fueron entrenados con 194.259 registros del departamento de Córdoba (2011–2022). '
    'Los resultados son orientativos — tu esfuerzo y preparación son el factor más importante.'
)

if not calcular and 'result' not in st.session_state:
    st.info('Completa tu perfil en el panel izquierdo y haz clic en **Calcular mi predicción**.')
    st.stop()

if calcular:
    with st.spinner('Calculando…'):
        prob_ing, score = _predict(profile, school_key)
    st.session_state['result'] = (prob_ing, score, dict(profile), school_key)

prob_ing, score, _prof, _school = st.session_state['result']
diff      = score - DEPT_MEAN
is_riesgo = prob_ing >= ING_THRESH


# ── Tabs ───────────────────────────────────────────────────────────────────────
tab_res, tab_mejora, tab_tips = st.tabs([
    'Mi resultado',
    'Como mejorar mi puntaje',
    'Consejos personalizados',
])


# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 — Resultado
# ════════════════════════════════════════════════════════════════════════════════
with tab_res:
    c1, c2, c3 = st.columns(3)
    c1.metric(
        'Puntaje Global Estimado',
        f'{score:.0f} pts',
        f'{diff:+.0f} vs. media departamental (241 pts)',
        delta_color='normal',
    )
    c2.metric(
        'Riesgo de Nivel A- en Inglés',
        f'{prob_ing:.1%}',
        'Alto riesgo — requiere refuerzo' if is_riesgo else 'Bajo riesgo — mantén el nivel',
        delta_color='inverse',
    )
    c3.metric(
        'Rango de confianza',
        f'{score - RMSE:.0f} – {score + RMSE:.0f} pts',
        f'±{RMSE:.0f} pts  (±1 RMSE del modelo)',
        delta_color='off',
    )

    st.markdown('---')
    col_g, col_i = st.columns([1, 1])

    with col_g:
        col_ing = '#b03a2e' if is_riesgo else '#2d6a4f'
        gauge = go.Figure(go.Indicator(
            mode='gauge+number',
            value=round(prob_ing * 100, 1),
            number={'suffix': '%', 'font': {'size': 34, 'color': col_ing}},
            title={'text': 'Probabilidad de nivel A- en inglés',
                   'font': {'size': 14, 'color': '#374151'}},
            gauge=dict(
                axis=dict(range=[0, 100], tickwidth=1,
                          tickcolor='#ccc', tickfont=dict(size=10)),
                bar=dict(color=col_ing, thickness=0.28),
                bgcolor='white', borderwidth=0,
                threshold=dict(
                    line=dict(color='#1e3a5f', width=3),
                    thickness=0.78,
                    value=round(ING_THRESH * 100, 1),
                ),
                steps=[
                    {'range': [0,               ING_THRESH * 100], 'color': '#f0fdf4'},
                    {'range': [ING_THRESH * 100, 100],             'color': '#fef2f2'},
                ],
            ),
        ))
        gauge.update_layout(
            height=290,
            margin=dict(l=20, r=20, t=50, b=10),
            paper_bgcolor='rgba(0,0,0,0)',
        )
        st.plotly_chart(gauge, use_container_width=True)

    with col_i:
        st.markdown('#### Qué significa tu resultado')
        if score >= DEPT_MEAN + 25:
            st.success(
                f'Tu puntaje estimado de **{score:.0f} pts** está notablemente **por encima** '
                f'del promedio departamental (241 pts). Estás en buen camino — mantén la '
                f'preparación y apunta a universidades de alta calidad.'
            )
        elif score >= DEPT_MEAN - 10:
            st.info(
                f'Tu puntaje estimado de **{score:.0f} pts** está **cerca** del promedio '
                f'departamental (241 pts). Con preparación focalizada en las áreas débiles '
                f'puedes subirlo de forma significativa antes del examen.'
            )
        else:
            st.warning(
                f'Tu puntaje estimado de **{score:.0f} pts** está **por debajo** del promedio '
                f'departamental (241 pts). El plan de mejora puede ayudarte a subirlo. '
                f'Hay condiciones en tu perfil que, si cambian, tienen un efecto importante.'
            )

        if is_riesgo:
            st.warning(
                f'Con una probabilidad de **{prob_ing:.0%}** de quedar en nivel A- en inglés, '
                f'esta es tu área más crítica. Dedicar 20–30 minutos diarios a inglés puede '
                f'marcar una diferencia real en tu puntaje final.'
            )
        else:
            st.success(
                f'Tu riesgo de nivel A- en inglés es bajo ({prob_ing:.0%}). '
                f'Mantén tu preparación en esta área para asegurar el resultado.'
            )

        st.caption(
            f'Intervalo de confianza del modelo: [{score-RMSE:.0f} – {score+RMSE:.0f}] pts. '
            f'El modelo explica el 37.8% de la varianza del puntaje. El 62.2% restante depende '
            f'de tu preparación, dedicación y desempeño el día del examen.'
        )


# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 — Plan de mejora
# ════════════════════════════════════════════════════════════════════════════════
with tab_mejora:
    st.markdown(
        '#### Cuánto podría subir tu puntaje si cambia cada condición'
    )
    st.caption(
        'Cada barra muestra el cambio estimado en tu puntaje global si solo esa variable '
        'mejora, manteniendo el resto de tu perfil igual. Úsalo como orientación, '
        'no como garantía — son promedios estadísticos.'
    )

    mejoras = []

    def _delta(key, val):
        p2 = dict(_prof)
        p2[key] = val
        _, s2 = _predict(p2, _school)
        return s2 - score

    if _prof['fami_tieneinternet'] == 'NO':
        mejoras.append(('Acceso a internet en el hogar', _delta('fami_tieneinternet', 'SI')))

    if _prof['fami_tienecomputador'] == 'NO':
        mejoras.append(('Computador en el hogar', _delta('fami_tienecomputador', 'SI')))

    if _prof['cole_bilingue'] == 'N':
        mejoras.append(('Colegio bilingüe', _delta('cole_bilingue', 'S')))

    if _prof['cole_jornada'] not in ['COMPLETA', 'MAÑANA']:
        mejoras.append(('Jornada completa o mañana', _delta('cole_jornada', 'COMPLETA')))

    if _prof['cole_naturaleza'] == 'OFICIAL':
        mejoras.append(('Colegio privado (no oficial)', _delta('cole_naturaleza', 'NO OFICIAL')))

    if _prof['fami_estratovivienda'] in ['ESTRATO 1', 'ESTRATO 2', 'SIN ESTRATO']:
        mejoras.append(('Mejora de estrato socioeconómico', _delta('fami_estratovivienda', 'ESTRATO 3')))

    if _prof['fami_educacionmadre'] in ['NINGUNO', 'PRIMARIA INCOMPLETA', 'PRIMARIA COMPLETA']:
        mejoras.append((
            'Madre con bachillerato completo',
            _delta('fami_educacionmadre', 'SECUNDARIA (BACHILLERATO) COMPLETA'),
        ))

    if not mejoras:
        st.success(
            'Tu perfil ya tiene las condiciones más favorables en los factores '
            'que los modelos pueden evaluar. Tu preparación individual es el '
            'factor clave a partir de aquí.'
        )
    else:
        mejoras.sort(key=lambda x: x[1], reverse=True)
        labels = [m[0] for m in mejoras]
        deltas = [m[1] for m in mejoras]
        colors = ['#2d6a4f' if d > 0 else '#b03a2e' for d in deltas]

        fig = go.Figure(go.Bar(
            x=deltas, y=labels,
            orientation='h',
            marker_color=colors,
            marker_line_width=0,
            text=[f'{d:+.1f} pts' for d in deltas],
            textposition='outside',
            textfont=dict(size=12),
            hovertemplate='%{y}<br>Cambio estimado: %{x:+.1f} pts<extra></extra>',
        ))
        fig.add_vline(x=0, line_color='#ccc', line_width=1.5)
        fig.update_layout(
            height=max(240, len(mejoras) * 68),
            margin=dict(l=10, r=80, t=20, b=10),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(title='Cambio estimado en puntaje global (pts)',
                       gridcolor='#efefef', linecolor='#efefef', zeroline=False),
            yaxis=dict(gridcolor='rgba(0,0,0,0)', linecolor='rgba(0,0,0,0)',
                       tickfont=dict(size=12)),
        )
        st.plotly_chart(fig, use_container_width=True)

        mejor = mejoras[0]
        if mejor[1] > 1:
            st.info(
                f'El mayor impacto potencial en tu caso es **{mejor[0]}**, '
                f'con un cambio estimado de **{mejor[1]:+.1f} pts**. '
                f'Enfoca los esfuerzos institucionales y familiares en esa dirección.'
            )


# ════════════════════════════════════════════════════════════════════════════════
# TAB 3 — Consejos personalizados
# ════════════════════════════════════════════════════════════════════════════════
with tab_tips:
    st.markdown('#### Consejos según tu perfil')

    tips = []

    if _prof['fami_tieneinternet'] == 'NO':
        tips.append(('warn', 'Prioridad: acceso a contenido digital',
            'El acceso a internet es uno de los predictores más relevantes del puntaje. '
            'Usa los puntos WiFi de tu municipio, la sala de informática de tu colegio o '
            'datos móviles para practicar con los simulacros gratuitos del ICFES '
            '(icfes.gov.co). Incluso 30 minutos diarios marcan diferencia.'))

    if _prof['fami_tienecomputador'] == 'NO':
        tips.append(('warn', 'Practica en pantalla antes del examen',
            'El Saber 11 se presenta en computador. Si no tienes uno en casa, '
            'usa los equipos del colegio o una biblioteca pública para familiarizarte '
            'con el formato de pantalla, el manejo del tiempo y la navegación entre preguntas.'))

    if _prof['cole_jornada'] in ['NOCHE', 'SABATINA']:
        tips.append(('warn', 'Refuerzo fuera del horario escolar',
            'Los estudiantes de jornada nocturna y sabatina tienen menos horas presenciales '
            'que sus pares. Dedica al menos 1.5 horas diarias adicionales a los cuadernillos '
            'de preparación ICFES. Prioriza Lectura Crítica y Matemáticas, que tienen el '
            'mayor peso en el puntaje global.'))

    if is_riesgo:
        tips.append(('warn', f'Inglés: área crítica — probabilidad A- del {prob_ing:.0%}',
            'El nivel A- es el más bajo del Marco Europeo. Para subirlo: usa apps gratuitas '
            '(Duolingo, BBC Learning English) 20 min/día, escucha podcasts en inglés básico, '
            'practica vocabulario con las guías ICFES. El vocabulario académico y la comprensión '
            'de lectura en inglés son las habilidades más evaluadas.'))

    if _prof['fami_estratovivienda'] in ['ESTRATO 1', 'ESTRATO 2', 'SIN ESTRATO']:
        tips.append(('info', 'Programas de apoyo disponibles para ti',
            'Como estudiante de estrato 1 o 2 tienes acceso prioritario a: créditos condonables '
            'ICETEX (Generación E), becas de la Gobernación de Córdoba y programas de preparación '
            'gratuitos del SENA. Habla con el orientador de tu colegio para conocer las '
            'convocatorias vigentes.'))

    if _prof['cole_area_ubicacion'] == 'RURAL':
        tips.append(('info', 'Recursos para colegios rurales',
            'El Ministerio de Educación tiene el Programa Todos a Aprender para colegios rurales '
            'y el ICFES publica materiales de preparación adaptados. Pregunta a tu rector si tu '
            'institución participa en estos programas y cómo acceder a los materiales.'))

    if score < DEPT_MEAN:
        tips.append(('warn', 'Matemáticas y Lectura Crítica son tu palanca más fuerte',
            f'Con tu puntaje estimado de {score:.0f} pts, mejorar en Matemáticas y Lectura Crítica '
            'tiene el mayor efecto en el total, porque son las dos áreas de mayor peso. '
            'Haz al menos un simulacro completo por semana y revisa cada error antes del examen.'))

    # Consejos siempre presentes
    tips.append(('info', 'Descarga los cuadernillos oficiales ICFES',
        'El ICFES publica guías de preparación gratuitas en icfes.gov.co para cada área. '
        'Haz mínimo 3 simulacros completos con control de tiempo antes del examen real, '
        'en condiciones lo más parecidas posible al día de la prueba.'))

    tips.append(('info', 'Plan de estudio semanal',
        'Divide la preparación en áreas: Lectura Crítica (lunes/jueves), Matemáticas (martes/viernes), '
        'Ciencias Naturales (miércoles), Inglés (diario, 20 min). Ciencias Sociales y '
        'Competencias Ciudadanas requieren menos preparación técnica — enfócate más en las primeras.'))

    tips.append(('info', 'El día del examen también importa',
        'Duerme bien los dos días previos, desayuna antes de entrar y llega con tiempo suficiente. '
        'Durante el examen, responde primero las preguntas que conoces y regresa a las difíciles. '
        'No dejes preguntas sin responder: no hay penalización por respuesta incorrecta.'))

    for kind, title, body in tips:
        cls = 'tip-card' if kind == 'info' else 'warn-card'
        st.markdown(
            f'<div class="{cls}">'
            f'<div class="tip-title">{title}</div>'
            f'<div class="tip-body">{body}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
