# 🛰️ Monitor Analítico de Telemetria Orbital (IA Preditiva)
### *FIAP - Global Solution 2026 | Generative AI For Engineering*

## 📌 Contexto do Problema
O espaço é um ambiente inóspito onde a latência de comunicação e a impossibilidade de manutenção física exigem que os satélites operem com alta autonomia. Este projeto resolve o desafio da **detecção precoce de falhas sistêmicas em satélites de órbita baixa (LEO)**. A solução utiliza Inteligência Artificial para correlacionar múltiplos atributos de telemetria em tempo real, prevendo colapsos estruturais ou elétricos antes que se tornem irreversíveis.

## 📊 Arquitetura e Dados
* **Dimensionalidade:** Embora a interface de simulação apresente 7 *sliders* interativos, o modelo **XGBoost** processa um tensor de **12 atributos** simultâneos. O pipeline injeta parâmetros espaciais fixos (Giroscópio X, Y, Z) e realiza *Feature Engineering* (Engenharia de Atributos) para calcular variáveis derivadas (`Potencia_Estimada_W` e `Magnitude_Giroscopio`) antes da predição.
* **Dataset:** 2.000 instâncias geradas sinteticamente com variações estocásticas baseadas em comportamento real de sensores aeroespaciais.

## 🚀 Funcionalidades Principais
* **Motor Preditivo (XGBoost):** Modelo de classificação de alta performance para identificação de padrões de falha.
* **Diagnóstico Neural Dinâmico:** Explicação justificada do porquê a IA classificou o estado como Nominal ou Crítico.
* **Raio-X de Subsistemas:** Monitoramento individualizado (Temperatura, Radiação, Bateria, etc.) com gradação de severidade (Estável/Atenção/Crítico).
* **Simulador de Cenários:** Botões para injeção de falhas controladas (Extremos) e cenários estocásticos (Aleatórios) para *stress test* do modelo.
* **Detector OOD (Out-of-Distribution):** Camada de segurança que identifica quando os dados de entrada fogem do histórico de treinamento, emitindo alertas de limitação de confiabilidade.

## 🛠️ Metodologia
1. **Pipeline de ML:** Estruturado em Jupyter Notebook (`pipeline_telemetria.ipynb`) abrangendo limpeza, treino e validação.
2. **Interpretabilidade (SHAP):** Aplicação da biblioteca SHAP para visualizar quais sensores exercem maior peso nas predições da IA.
3. **Deploy:** Dashboard interativo em **Streamlit**, focado em baixa latência e UX voltada para operação de missão.

## ⚙️ Instruções de Execução

### Pré-requisitos
* Python 3.11+

### Configuração
1. Clone o repositório:
   ```bash
   git clone [https://github.com/GuiMilheiro/Generative-AI-Global-Solution-2026.1](https://github.com/GuiMilheiro/Generative-AI-Global-Solution-2026.1)
   ```
2. Configure o ambiente virtual e instale as dependências:
   ```bash
    python -m venv venv

    # Ativar ambiente (Windows)
    .\venv\Scripts\activate

    # Instalar dependências
    pip install pandas numpy scikit-learn xgboost shap matplotlib streamlit joblib
    ```
3. Treinamento: Rode o notebook 

    `pipeline_telemetria.ipynb` 
    
    para gerar o modelo 
    
    `modelo_xgboost_telemetria.pkl`

4. Interface: Inicie o Dashboard Web:
    
    `streamlit run app.py`
    
   
🔗 Acesso à Aplicação
Acesse o Dashboard Interativo Hospedado no Streamlit Cloud: https://generative-ai-global-solution-20261-rjuvsl6gn5tajzgt4afvtg.streamlit.app/

Projeto desenvolvido por:

- Guilherme Dejulio Milheiro (RM550295)
- Enzo Vasconcelos (RM550702)
- Ricardo Queiroz (RM94241)
- Jhonatan Curci (RM94188)
- Felipe Hideki (RM98323) 

como requisito da Global Solution 2026 - FIAP.
