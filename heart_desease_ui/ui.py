import joblib
import pandas as pd
import streamlit as st
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent


st.set_page_config(
    page_title="Risco Cardiaco",
    page_icon=":heart:",
    layout="wide",
)


@st.cache_data
def carregar_dados():
    return pd.read_parquet(BASE_DIR / "heart_disease_risk_dataset.parquet")


@st.cache_resource
def carregar_modelo():
    return joblib.load(BASE_DIR / "pipeline.pkl")


df = carregar_dados()
features = df.drop(columns=["patient_id", "has_heart_disease"])
modelo = carregar_modelo()

opcoes_sexo = {
    "Masculino": "Male",
    "Feminino": "Female",
}

opcoes_dor_peito = {
    "Assintomatica": "Asymptomatic",
    "Dor nao anginosa": "Non-Anginal Pain",
    "Angina atipica": "Atypical Angina",
    "Angina tipica": "Typical Angina",
}

opcoes_fumante = {
    "Nunca fumou": "Never",
    "Ex-fumante": "Former",
    "Fumante atual": "Current",
}

st.markdown(
    """
    <style>
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2.5rem;
            max-width: 1180px;
        }

        .hero {
            border: 1px solid rgba(148, 163, 184, 0.25);
            border-radius: 8px;
            padding: 1.5rem 1.6rem;
            background: rgba(148, 163, 184, 0.08);
            margin-bottom: 1rem;
        }

        .hero h1 {
            font-size: 2rem;
            line-height: 1.15;
            margin: 0 0 .4rem 0;
            letter-spacing: 0;
        }

        .hero p {
            color: rgba(120, 120, 120, 0.95);
            font-size: 1rem;
            margin: 0;
            max-width: 820px;
        }

        .notice {
            border-left: 4px solid #2563eb;
            background: rgba(37, 99, 235, 0.08);
            padding: .85rem 1rem;
            border-radius: 6px;
            margin: 1rem 0 1.3rem 0;
        }

        .section-title {
            font-size: 1.08rem;
            font-weight: 700;
            margin: .35rem 0 .2rem 0;
        }

        div[data-testid="stForm"] {
            border: 1px solid rgba(148, 163, 184, 0.25);
            border-radius: 8px;
            padding: 1.15rem 1.25rem 1.25rem 1.25rem;
            background: rgba(148, 163, 184, 0.05);
        }

        div[data-testid="stMetric"] {
            border: 1px solid rgba(148, 163, 184, 0.22);
            border-radius: 8px;
            padding: .85rem 1rem;
            background: rgba(148, 163, 184, 0.05);
        }

        .result-card {
            border-radius: 8px;
            padding: 1.1rem 1.2rem;
            border: 1px solid rgba(148, 163, 184, 0.25);
            background: rgba(148, 163, 184, 0.06);
        }

        .result-card h3 {
            margin: 0 0 .3rem 0;
            letter-spacing: 0;
        }

        .result-card p {
            margin: 0;
            color: rgba(120, 120, 120, 0.95);
        }

        .stButton > button {
            border-radius: 6px;
            font-weight: 700;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


aba_previsao, aba_metricas = st.tabs(
    [
        "Previsao",
        "Metricas de treino",
    ]
)

with aba_previsao:
    st.markdown(
        """
        <div class="hero">
            <h1>Previsao de risco cardiaco</h1>
            <p>
                Modelo de Regressao Logistica treinado com dados normalizados para estimar
                a probabilidade de doenca cardiaca a partir de indicadores clinicos e de estilo de vida.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    resumo_1, resumo_2, resumo_3 = st.columns(3)
    resumo_1.metric("Amostras no treino", f"{len(df):,}".replace(",", "."))
    resumo_2.metric("Variaveis analisadas", len(features.columns))
    resumo_3.metric("Tipo de modelo", "Regressao Logistica")

    st.markdown(
        """
        <div class="notice">
            Este resultado tem finalidade academica e nao substitui avaliacao, diagnostico
            ou orientacao de um profissional de saude.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.link_button(
        label="Dataset utilizado no treinamento",
        url="https://www.kaggle.com/datasets/srisyra02/heart-disease-risk-2026",
    )

    with st.form(key="predict_model"):
        st.markdown('<div class="section-title">Dados do paciente</div>', unsafe_allow_html=True)
        paciente_col_1, paciente_col_2, paciente_col_3 = st.columns(3)
        with paciente_col_1:
            age = st.number_input("Idade", min_value=18, max_value=100, value=54, step=1)
        with paciente_col_2:
            sex_label = st.selectbox("Sexo", list(opcoes_sexo.keys()))
        with paciente_col_3:
            family_history = st.checkbox("Historico familiar de doenca cardiaca")

        st.divider()
        st.markdown('<div class="section-title">Sinais vitais e exames</div>', unsafe_allow_html=True)
        sinais_col_1, sinais_col_2, sinais_col_3 = st.columns(3)
        with sinais_col_1:
            resting_bp_systolic = st.number_input(
                "Pressao sistolica em repouso (mmHg)",
                min_value=70,
                max_value=220,
                value=128,
                step=1,
            )
            cholesterol_total = st.number_input(
                "Colesterol total (mg/dL)",
                min_value=80,
                max_value=400,
                value=189,
                step=1,
            )
            fasting_blood_sugar = st.number_input(
                "Glicemia em jejum (mg/dL)",
                min_value=50,
                max_value=260,
                value=119,
                step=1,
            )
        with sinais_col_2:
            resting_bp_diastolic = st.number_input(
                "Pressao diastolica em repouso (mmHg)",
                min_value=40,
                max_value=140,
                value=81,
                step=1,
            )
            hdl = st.number_input("HDL (mg/dL)", min_value=10, max_value=140, value=55, step=1)
            hba1c = st.number_input(
                "Hemoglobina glicada HbA1c (%)",
                min_value=3.0,
                max_value=12.0,
                value=5.8,
                step=0.1,
                format="%.1f",
            )
        with sinais_col_3:
            bmi = st.number_input("IMC", min_value=12.0, max_value=60.0, value=25.3, step=0.1, format="%.1f")
            ldl = st.number_input("LDL (mg/dL)", min_value=20, max_value=260, value=103, step=1)
            triglycerides = st.number_input(
                "Triglicerideos (mg/dL)",
                min_value=20,
                max_value=600,
                value=152,
                step=1,
            )

        st.divider()
        st.markdown('<div class="section-title">Cardiologia</div>', unsafe_allow_html=True)
        cardio_col_1, cardio_col_2, cardio_col_3 = st.columns(3)
        with cardio_col_1:
            resting_heart_rate = st.number_input(
                "Frequencia cardiaca em repouso (bpm)",
                min_value=35,
                max_value=140,
                value=81,
                step=1,
            )
        with cardio_col_2:
            max_heart_rate_achieved = st.number_input(
                "Frequencia cardiaca maxima atingida (bpm)",
                min_value=60,
                max_value=230,
                value=165,
                step=1,
            )
        with cardio_col_3:
            st_depression = st.number_input(
                "Depressao do segmento ST",
                min_value=0.0,
                max_value=8.0,
                value=1.0,
                step=0.1,
                format="%.1f",
            )

        cardio_col_4, cardio_col_5 = st.columns([2, 1])
        with cardio_col_4:
            chest_pain_label = st.selectbox("Tipo de dor no peito", list(opcoes_dor_peito.keys()))
        with cardio_col_5:
            exercise_induced_angina = st.checkbox("Angina induzida por exercicio")

        st.divider()
        st.markdown('<div class="section-title">Estilo de vida</div>', unsafe_allow_html=True)
        estilo_col_1, estilo_col_2, estilo_col_3 = st.columns(3)
        with estilo_col_1:
            smoker_label = st.selectbox("Historico de tabagismo", list(opcoes_fumante.keys()))
            exercise_minutes_per_week = st.number_input(
                "Exercicio por semana (min)",
                min_value=0,
                max_value=700,
                value=140,
                step=10,
            )
            wearable_owner = st.checkbox("Usa smartwatch ou pulseira")
        with estilo_col_2:
            alcohol_units_per_week = st.number_input(
                "Unidades de alcool por semana",
                min_value=0.0,
                max_value=80.0,
                value=5.7,
                step=0.5,
                format="%.1f",
            )
            sleep_hours = st.number_input(
                "Horas de sono por noite",
                min_value=0.0,
                max_value=14.0,
                value=7.0,
                step=0.5,
                format="%.1f",
            )
        with estilo_col_3:
            daily_steps = st.number_input(
                "Passos diarios",
                min_value=0,
                max_value=30000,
                value=6200,
                step=250,
            )
            stress_score = st.slider("Nivel de estresse", min_value=0.0, max_value=100.0, value=48.0, step=1.0)
            diet_quality_score = st.slider(
                "Qualidade da dieta",
                min_value=0.0,
                max_value=100.0,
                value=59.0,
                step=1.0,
            )

        enviar_formulario = st.form_submit_button("Calcular risco", use_container_width=True)

    if enviar_formulario:
        campos_formulario = {
            "age": age,
            "sex": opcoes_sexo[sex_label],
            "resting_bp_systolic": resting_bp_systolic,
            "resting_bp_diastolic": resting_bp_diastolic,
            "cholesterol_total": cholesterol_total,
            "hdl": hdl,
            "ldl": ldl,
            "triglycerides": triglycerides,
            "fasting_blood_sugar": fasting_blood_sugar,
            "hba1c": hba1c,
            "bmi": bmi,
            "resting_heart_rate": resting_heart_rate,
            "max_heart_rate_achieved": max_heart_rate_achieved,
            "chest_pain_type": opcoes_dor_peito[chest_pain_label],
            "exercise_induced_angina": exercise_induced_angina,
            "st_depression": st_depression,
            "family_history": family_history,
            "smoker_status": opcoes_fumante[smoker_label],
            "alcohol_units_per_week": alcohol_units_per_week,
            "exercise_minutes_per_week": exercise_minutes_per_week,
            "sleep_hours": sleep_hours,
            "stress_score": stress_score,
            "wearable_owner": wearable_owner,
            "daily_steps": daily_steps,
            "diet_quality_score": diet_quality_score,
        }

        entrada = pd.DataFrame([campos_formulario], columns=features.columns)
        previsao = int(modelo.predict(entrada)[0])
        risco = None

        if hasattr(modelo, "predict_proba"):
            risco = float(modelo.predict_proba(entrada)[0][1])

        resultado_col_1, resultado_col_2 = st.columns([1.2, 2])
        with resultado_col_1:
            if risco is not None:
                st.metric("Probabilidade estimada", f"{risco:.1%}")
                st.progress(min(max(risco, 0.0), 1.0))
            else:
                st.metric("Classificacao", "Risco alto" if previsao == 1 else "Risco baixo")

        with resultado_col_2:
            if previsao == 1:
                st.markdown(
                    """
                    <div class="result-card">
                        <h3>Resultado: risco elevado</h3>
                        <p>
                            O perfil informado foi classificado como maior risco pelo modelo.
                            Use este retorno apenas como apoio academico e procure avaliacao profissional
                            para qualquer decisao de saude.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    """
                    <div class="result-card">
                        <h3>Resultado: risco nao elevado</h3>
                        <p>
                            O perfil informado nao foi classificado como alto risco pelo modelo.
                            Mesmo assim, acompanhamento medico e habitos preventivos continuam importantes.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        with st.expander("Ver dados enviados ao modelo"):
            st.dataframe(entrada, use_container_width=True, hide_index=True)


with aba_metricas:
    st.subheader("Amostra do dataset original")
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.divider()

    grafico_col_1, grafico_col_2 = st.columns(2)
    with grafico_col_1:
        st.subheader("Correlacao das features")
        st.image(str(BASE_DIR / "plots/correlacao.png"), use_container_width=True)

        st.subheader("Precision-Recall")
        st.image(str(BASE_DIR / "plots/ap.png"), use_container_width=True)

    with grafico_col_2:
        st.subheader("Curva ROC")
        st.image(str(BASE_DIR / "plots/roc_auc.png"), use_container_width=True)

        st.subheader("Importancia das features")
        st.image(str(BASE_DIR / "plots/importancia_features_target.png"), use_container_width=True)
