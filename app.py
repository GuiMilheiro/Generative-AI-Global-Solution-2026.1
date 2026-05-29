import streamlit as st
import pandas as pd
import numpy as np
import joblib
import random

# 1. Configuração da página
st.set_page_config(page_title="Monitor de Telemetria Espacial", layout="wide")

# 2. CSS Customizado
st.markdown("""
    <style>
           .block-container {
                padding-top: 1.5rem;
                padding-bottom: 1.5rem;
            }
    </style>
    """, unsafe_allow_html=True)

st.title("🛰️ Monitor de Anomalias em Órbita")
# TEXTO CORRIGIDO PARA NÃO CAUSAR CONFUSÃO:
st.write("O motor de Inteligência Artificial avalia a correlação entre os sensores ativos e variáveis derivadas em tempo real.")

@st.cache_resource
def load_model():
    return joblib.load('modelo_xgboost_telemetria.pkl')

model = load_model()

if 'temp' not in st.session_state:
    st.session_state.temp = 15.0
    st.session_state.rad = 5.0
    st.session_state.volt = 24.0
    st.session_state.corr = 5.0
    st.session_state.cpu = 50.0
    st.session_state.mem = 50.0
    st.session_state.lat = 250.0

# --- SIDEBAR ---
st.sidebar.header("🕹️ Centro de Comando")
st.sidebar.info("Injete dados simulados no sistema para testar o modelo preditivo.")

if st.sidebar.button("🚨 Simular Falha Crítica", use_container_width=True, help="Injeta valores extremos na telemetria (ex: superaquecimento ou bateria crítica) para forçar o colapso do sistema e demonstrar o alerta da IA."):
    st.session_state.temp = round(random.uniform(10.0, 25.0), 2)
    st.session_state.rad = round(random.uniform(1.0, 8.0), 2)
    st.session_state.volt = round(random.uniform(23.0, 26.0), 2)
    st.session_state.corr = round(random.uniform(4.0, 8.0), 2)
    st.session_state.cpu = round(random.uniform(20.0, 50.0), 2)
    st.session_state.mem = round(random.uniform(20.0, 50.0), 2)
    st.session_state.lat = round(random.uniform(150.0, 250.0), 2)

    cenario = random.choice(['energia', 'termico', 'comunicacao', 'caos_total'])
    if cenario == 'energia':
        st.session_state.volt = round(random.uniform(14.0, 18.0), 2)
        st.session_state.cpu = round(random.uniform(92.0, 100.0), 2)
    elif cenario == 'termico':
        st.session_state.temp = round(random.uniform(85.0, 140.0), 2)
        st.session_state.rad = round(random.uniform(30.0, 50.0), 2)
    elif cenario == 'comunicacao':
        st.session_state.lat = round(random.uniform(650.0, 1000.0), 2)
    elif cenario == 'caos_total':
        st.session_state.volt = round(random.uniform(14.0, 18.0), 2)
        st.session_state.temp = round(random.uniform(85.0, 140.0), 2)
        st.session_state.lat = round(random.uniform(650.0, 1000.0), 2)

if st.sidebar.button("🎲 Simular Situação Aleatória", use_container_width=True, help="Sorteia valores completamente aleatórios para os sensores. Testa a capacidade da IA de analisar zonas cinzentas e cenários imprevisíveis."):
    st.session_state.temp = round(random.uniform(-10.0, 100.0), 2)
    st.session_state.rad = round(random.uniform(0.0, 35.0), 2)
    st.session_state.volt = round(random.uniform(18.0, 28.0), 2)
    st.session_state.corr = round(random.uniform(0.0, 12.0), 2)
    st.session_state.cpu = round(random.uniform(10.0, 95.0), 2)
    st.session_state.mem = round(random.uniform(10.0, 95.0), 2)
    st.session_state.lat = round(random.uniform(100.0, 600.0), 2)

st.sidebar.divider()
st.sidebar.markdown("### 📖 Manual de Simulação")
st.sidebar.error("**🚨 Falha Crítica:** Injeta valores extremos na telemetria para forçar o colapso sistêmico e demonstrar o bloqueio de segurança da IA.")
st.sidebar.info("**🎲 Aleatório:** Sorteia dados imprevisíveis para testar a capacidade do modelo em lidar com zonas cinzentas da operação orbital.")
st.sidebar.divider()

# --- ÁREA PRINCIPAL ---
st.subheader("🎛️ Painel de Sensores Ativos")
col_a, col_b, col_c, col_d = st.columns(4)

with col_a:
    temp = st.slider("Temperatura (°C)", -50.0, 150.0, key="temp")
    rad = st.slider("Radiação (mSv)", 0.0, 50.0, key="rad")
with col_b:
    volt = st.slider("Bateria (V)", 15.0, 30.0, key="volt")
    corrente = st.slider("Corrente Solar (A)", 0.0, 15.0, key="corr")
with col_c:
    cpu = st.slider("Carga CPU (%)", 0.0, 100.0, key="cpu")
    mem = st.slider("Uso Memória (%)", 0.0, 100.0, key="mem")
with col_d:
    lat = st.slider("Latência (ms)", 0.0, 1000.0, key="lat")
    
    potencia = volt * corrente
    mag_giro = np.sqrt(0.1**2 + 0.1**2 + 0.1**2)

st.divider()

# --- ÁREA DE RESULTADOS ---
col_diag1, col_diag2 = st.columns([1, 1.5]) 

with col_diag1:
    if st.button("Executar Diagnóstico Neural", type="primary", use_container_width=True):
        input_data = pd.DataFrame({
            'Temperatura_C': [temp], 'Radiacao_mSv': [rad], 'Voltagem_Bateria_V': [volt],
            'Corrente_Painel_Solar_A': [corrente], 'Giroscopio_X_rads': [0.1],
            'Giroscopio_Y_rads': [0.1], 'Giroscopio_Z_rads': [0.1],
            'Carga_CPU_Pct': [cpu], 'Uso_Memoria_Pct': [mem],
            'Latencia_Comunicacao_ms': [lat], 'Potencia_Estimada_W': [potencia],
            'Magnitude_Giroscopio': [mag_giro]
        })
        
        with st.spinner('Modelos neurais processando dados...'):
            prediction = model.predict(input_data)[0]
            
        if prediction == 1:
            st.error("⚠️ PREDIÇÃO DA IA: Falha Sistêmica Iminente!")
        else:
            st.success("✅ PREDIÇÃO DA IA: Satélite Operacional.")

        st.markdown("### 📊 Raio-X Individual dos Sensores")
        rx_col1, rx_col2 = st.columns(2)
        
        with rx_col1:
            if temp >= 80: st.markdown("🔴 **Temp:** CRÍTICO")
            elif temp >= 50: st.markdown("🟡 **Temp:** ATENÇÃO")
            else: st.markdown("🟢 **Temp:** ESTÁVEL")
            
            if rad >= 30: st.markdown("🔴 **Radiação:** CRÍTICO")
            elif rad >= 15: st.markdown("🟡 **Radiação:** ATENÇÃO")
            else: st.markdown("🟢 **Radiação:** ESTÁVEL")
            
            if cpu >= 85: st.markdown("🔴 **CPU:** CRÍTICO")
            elif cpu >= 70: st.markdown("🟡 **CPU:** ATENÇÃO")
            else: st.markdown("🟢 **CPU:** ESTÁVEL")

            if mem >= 85: st.markdown("🔴 **Memória:** CRÍTICO")
            elif mem >= 70: st.markdown("🟡 **Memória:** ATENÇÃO")
            else: st.markdown("🟢 **Memória:** ESTÁVEL")

        with rx_col2:
            if volt <= 21: st.markdown("🔴 **Bateria:** CRÍTICO")
            elif volt <= 23: st.markdown("🟡 **Bateria:** ATENÇÃO")
            else: st.markdown("🟢 **Bateria:** ESTÁVEL")
            
            if corrente <= 2: st.markdown("🔴 **Corrente:** CRÍTICO")
            elif corrente <= 4: st.markdown("🟡 **Corrente:** ATENÇÃO")
            else: st.markdown("🟢 **Corrente:** ESTÁVEL")
            
            if lat >= 400: st.markdown("🔴 **Latência:** CRÍTICO")
            elif lat >= 250: st.markdown("🟡 **Latência:** ATENÇÃO")
            else: st.markdown("🟢 **Latência:** ESTÁVEL")

with col_diag2:
    if 'prediction' in locals():
        st.markdown("### 🧠 Pensamento Analítico da IA")
        
        # Mapeando os problemas ativos para gerar o texto dinâmico
        problemas_ativos = []
        if temp >= 80: problemas_ativos.append("Superaquecimento")
        if rad >= 30: problemas_ativos.append("Alta Radiação")
        if volt <= 21: problemas_ativos.append("Bateria Crítica")
        if corrente <= 2: problemas_ativos.append("Baixa Corrente Solar")
        if cpu >= 85: problemas_ativos.append("Sobrecarga de CPU")
        if mem >= 85: problemas_ativos.append("Esgotamento de Memória")
        if lat >= 400: problemas_ativos.append("Perda de Comunicação")

        # Formatando a lista de problemas em uma string legível
        if len(problemas_ativos) > 1:
            texto_problemas = ", ".join(problemas_ativos[:-1]) + " e " + problemas_ativos[-1]
        elif len(problemas_ativos) == 1:
            texto_problemas = problemas_ativos[0]
        else:
            texto_problemas = "uma combinação complexa de estresse nos componentes"

        algum_sensor_critico = len(problemas_ativos) > 0
        
        if prediction == 0 and algum_sensor_critico:
            st.info(f"""
            **Por que o Status da IA é Verde apesar de sensores no nível Crítico?**
            
            O modelo XGBoost não avalia os sensores de forma isolada. 
            Ele analisa a **correlação estatística** de todas as variáveis simultaneamente. 
            
            Embora o Raio-X indique problemas graves (como: {texto_problemas}), a IA aprendeu 
            em seu treinamento que uma falha fatal exige uma combinação específica de fatores. 
            Como os outros subsistemas estão compensando a anomalia, 
            o algoritmo calculou que o satélite consegue manter a integridade orbital.
            """)
        elif prediction == 1:
            st.warning(f"""
            **Justificativa para Bloqueio de Segurança:**
            
            O algoritmo encontrou uma correlação matemática letal entre os dados, 
            com destaque para a presença de **{texto_problemas}**. Historicamente, 
            essa assinatura específica de degradação cruzada compromete os sistemas vitais. 
            A previsão é de colapso irreversível caso o *Safe Mode* não seja ativado imediatamente.
            """)
        elif len(problemas_ativos) >= 3:
            st.warning(f"""
            **⚠️ Alerta de Limitação do Modelo (Out-of-Distribution):**
            
            O status da IA retornou Verde, mas o Raio-X detectou múltiplas falhas graves (como: {texto_problemas}).
            
            **Diagnóstico Técnico:** Isso é um fenômeno de Machine Learning conhecido como 
            dados *Out-of-Distribution*. O modelo XGBoost baseia suas decisões puramente no 
            histórico que aprendeu. Essa combinação exata de extremos provavelmente nunca ocorreu nos dados de treinamento. 
            Como a IA não tem "senso comum", ela falhou ao classificar esse cenário inédito. Uma trava de segurança baseada em 
            heurística deve sobrepor a IA neste momento.
            """)
        else:
            st.success("""
            **Justificativa de Estabilidade:**
            
            A matriz de dados orbitais não apresenta desvios padrão perigosos.
            A IA não encontrou nenhuma assinatura estatística de falha iminente na comunicação cruzada dos sensores.
            """)