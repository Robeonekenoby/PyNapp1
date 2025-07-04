import streamlit as st
import pandas as pd
import numpy as np
import pdfplumber
import plotly.express as px
import os
from scipy import stats
import streamlit as st
from st_paywall import add_auth


st.set_page_config(page_title="Data Analyzer", layout="wide")

st.title("📊 Caricamento, Analisi e Data Storytelling con PyNarrative")

add_auth(required=True)

# Dopo l'autenticazione

st.write(f"Subscription Status: {st.session_state.user_subscribed}")
st.write("🎉 Evviva! Tutto ok e sei iscritto!")
st.write(f'A proposito, la tua email è: {st.session_state.email}')

# Funzione di caricamento file
def load_data(file):
    name = file.name.lower()
    ext = os.path.splitext(name)[-1]
    try:
        if ext in [".csv", ".tsv"]:
            return pd.read_csv(file, sep=None, engine='python')
        elif ext in [".xlsx", ".xls"]:
            return pd.read_excel(file)
        elif ext == ".json":
            return pd.read_json(file)
        elif ext == ".parquet":
            return pd.read_parquet(file)
        elif ext == ".feather":
            return pd.read_feather(file)
        elif ext == ".html":
            return pd.read_html(file)[0]
        elif ext == ".pdf":
            with pdfplumber.open(file) as pdf:
                for page in pdf.pages:
                    table = page.extract_table()
                    if table and len(table) > 1:
                        return pd.DataFrame(table[1:], columns=table[0])
    except Exception as e:
        st.error(f"Errore nel caricamento: {e}")
    return None

# Funzione di pulizia
def clean_data(df):
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    df.dropna(how='all', inplace=True)
    df.dropna(axis=1, how='all', inplace=True)
    df.drop_duplicates(inplace=True)
    for col in df.select_dtypes(include='object').columns:
        try:
            df[col] = pd.to_numeric(df[col], errors='ignore')
        except:
            pass
        try:
            df[col] = pd.to_datetime(df[col], errors='ignore')
        except:
            pass
    return df

# Regola del 30% di Malizia per affidabilità della media
def malizia_30_percent_rule(df):
    """
    Regola del 30% del Prof. Malizia:
    Se std < 30% della media → media affidabile
    Se std >= 30% della media → meglio usare mediana
    """
    results = {}
    for col in df.select_dtypes(include=np.number).columns:
        if df[col].count() > 0:  # Evita divisione per zero
            mean_val = df[col].mean()
            std_val = df[col].std()
            if mean_val != 0:  # Evita divisione per zero
                std_percent = (std_val / abs(mean_val)) * 100
                is_reliable = std_percent < 30
                results[col] = {
                    'mean': round(mean_val, 4),
                    'std': round(std_val, 4),
                    'std_percent': round(std_percent, 2),
                    'mean_reliable': is_reliable,
                    'recommended': 'Media' if is_reliable else 'Mediana',
                    'median': round(df[col].median(), 4)
                }
            else:
                results[col] = {
                    'mean': 0,
                    'std': round(std_val, 4),
                    'std_percent': float('inf'),
                    'mean_reliable': False,
                    'recommended': 'Mediana',
                    'median': round(df[col].median(), 4)
                }
    return results

# Test di normalità e asimmetria (Fischer)
def normality_analysis(df):
    """
    Analisi della normalità secondo Fischer:
    - Kurtosis ≈ 0 → distribuzione normale
    - Asimmetria tra -0.5 e 0.5 → dati simmetrici
    - Asimmetria tra -1/-0.5 e 0.5/1 → moderatamente distorti
    - Asimmetria < -1 o > 1 → molto distorti
    """
    results = {}
    for col in df.select_dtypes(include=np.number).columns:
        if df[col].count() > 2:  # Serve almeno 3 valori
            skewness = df[col].skew()
            kurt = df[col].kurtosis()
            
            # Classificazione asimmetria
            if -0.5 <= skewness <= 0.5:
                skew_class = "Simmetrici"
                skew_color = "🟢"
            elif -1 <= skewness < -0.5 or 0.5 < skewness <= 1:
                skew_class = "Moderatamente distorti"
                skew_color = "🟡"
            else:
                skew_class = "Molto distorti"
                skew_color = "🔴"
            
            # Classificazione kurtosis (Fischer)
            if abs(kurt) < 0.5:
                kurt_class = "Normale (Fischer)"
                kurt_color = "🟢"
            elif abs(kurt) < 1:
                kurt_class = "Quasi normale"
                kurt_color = "🟡"
            else:
                kurt_class = "Non normale"
                kurt_color = "🔴"
            
            results[col] = {
                'skewness': round(skewness, 4),
                'kurtosis': round(kurt, 4),
                'skew_classification': skew_class,
                'skew_color': skew_color,
                'kurt_classification': kurt_class,
                'kurt_color': kurt_color,
                'is_normal': abs(kurt) < 0.5 and abs(skewness) <= 0.5
            }
    return results

# Suggerimento automatico per metodo di correlazione
def suggest_correlation_method(df, outlier_info):
    """
    Suggerisce il metodo di correlazione basato su:
    - Normalità dei dati → Pearson
    - Dati non normali → Spearman
    - Molti outlier → Kendall Tau
    """
    normality = normality_analysis(df)
    suggestions = {}
    
    for col in df.select_dtypes(include=np.number).columns:
        if col in normality and col in outlier_info:
            is_normal = normality[col]['is_normal']
            outlier_percent = outlier_info[col]['percentage']
            
            if outlier_percent > 15:  # Molti outlier
                method = "Kendall Tau"
                reason = f"Molti outlier ({outlier_percent}%)"
                color = "🔴"
            elif is_normal:
                method = "Pearson"
                reason = "Dati normali"
                color = "🟢"
            else:
                method = "Spearman"
                reason = "Dati non normali"
                color = "🟡"
            
            suggestions[col] = {
                'method': method,
                'reason': reason,
                'color': color
            }
    
    return suggestions

# Statistiche numeriche avanzate
def describe_numeric_advanced(df):
    desc = df.describe().T
    desc['median'] = df.median(numeric_only=True)
    desc['iqr'] = df.quantile(0.75) - df.quantile(0.25)
    desc['missing'] = df.isnull().sum()
    
    # Aggiungi regola del 30% di Malizia
    malizia_results = malizia_30_percent_rule(df)
    desc['std_percent'] = [malizia_results.get(col, {}).get('std_percent', 0) for col in desc.index]
    desc['recommended_stat'] = [malizia_results.get(col, {}).get('recommended', 'N/A') for col in desc.index]
    
    return desc

# Rilevamento outlier
def detect_outliers(df):
    outliers = {}
    for col in df.select_dtypes(include=np.number).columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        mask = (df[col] < lower) | (df[col] > upper)
        outliers[col] = {
            "count": mask.sum(),
            "percentage": round(mask.mean() * 100, 2),
            "bounds": (round(lower, 2), round(upper, 2))
        }
    return outliers

# === GUIDA UNIFICATA COMPLETA ===
with st.expander("📖 GUIDA COMPLETA ALL'ANALISI DATI - Prof. Malizia", expanded=False):
    st.markdown("""
    # 🎯 **LA GUIDA DEFINITIVA ALL'ANALISI STATISTICA**
    
    *Tutte le regole, i metodi e le interpretazioni in un unico posto*
    
    ---
    
    ## 📏 **1. REGOLA DEL 30% (Prof. Malizia) - QUALE STATISTICA USARE?**
    
    ### 🤔 **Il Problema:**
    Quando hai una serie di numeri, quale valore rappresenta meglio il "centro" dei dati?
    - La **media** (somma diviso numero elementi)?
    - La **mediana** (valore che sta nel mezzo)?
    
    ### 🎯 **La Soluzione - Regola del 30%:**
    
    **Se `Deviazione Standard / Media < 30%` → USA LA MEDIA ✅**
    
    **Se `Deviazione Standard / Media ≥ 30%` → USA LA MEDIANA ✅**
    
    ### 💡 **Perché funziona?**
    - **Deviazione Standard alta** = i dati sono molto sparsi
    - **Media bassa con deviazione alta** = ci sono valori estremi che "tirano" la media
    - **La mediana è immune** ai valori estremi
    
    ### 📊 **Esempi Pratici:**
    - **Stipendi azienda**: Media = 50k€, Std = 40k€ → 40/50 = 80% > 30% → **USA MEDIANA**
    - **Voti esame**: Media = 24, Std = 3 → 3/24 = 12.5% < 30% → **USA MEDIA**
    - **Prezzi case**: Media = 300k€, Std = 200k€ → 200/300 = 67% > 30% → **USA MEDIANA**
    
    ---
    
    ## 📊 **2. TEST DI NORMALITÀ (Fischer) - I TUOI DATI SONO "NORMALI"?**
    
    ### 🤔 **Perché è importante?**
    Prima di scegliere test statistici o correlazioni, devi sapere se i tuoi dati seguono una **distribuzione normale** (la famosa "curva a campana").
    
    ### 🔍 **Due Indicatori Chiave:**
    
    #### **A) SKEWNESS (Asimmetria) - La distribuzione è bilanciata?**
    
    **🎯 Cosa misura:** Quanto la distribuzione è "sbilanciata" verso sinistra o destra.
    
    **📏 Scale di giudizio:**
    - **`[-0.5, 0.5]`** = 🟢 **SIMMETRICA** (quasi perfetta)
    - **`[-1, -0.5]` o `[0.5, 1]`** = 🟡 **MODERATAMENTE DISTORTA**
    - **`< -1` o `> 1`** = 🔴 **MOLTO DISTORTA** (problematica)
    
    **💭 Come interpretare:**
    - **Skewness = 0**: Perfettamente simmetrica (campana perfetta)
    - **Skewness > 0**: Coda lunga verso DESTRA (es. stipendi - pochi molto ricchi)
    - **Skewness < 0**: Coda lunga verso SINISTRA (es. voti - pochi molto bassi)
    
    #### **B) KURTOSIS (Curtosi) - La distribuzione è "appuntita"?**
    
    **🎯 Cosa misura:** Quanto la distribuzione è concentrata intorno al centro.
    
    **📏 Regola di Fischer:**
    - **`|Kurtosis| < 0.5`** = 🟢 **NORMALE** (quasi perfetta)
    - **`|Kurtosis| < 1`** = 🟡 **ACCETTABILE**
    - **`|Kurtosis| ≥ 1`** = 🔴 **NON NORMALE**
    
    **💭 Come interpretare:**
    - **Kurtosis = 0**: Distribuzione normale perfetta
    - **Kurtosis > 0**: Più "APPUNTITA" (molti valori vicini alla media + outlier estremi)
    - **Kurtosis < 0**: Più "PIATTA" (valori distribuiti più uniformemente)
    
    ### ✅ **QUANDO I DATI SONO "NORMALI"?**
    
    Una distribuzione è **NORMALE** quando:
    1. **Skewness** è tra `-0.5` e `0.5` (simmetrica)
    2. **Kurtosis** è vicina a `0` (secondo Fischer)
    
    ---
    
    ## 🔗 **3. SCELTA DEL METODO DI CORRELAZIONE - QUALE USARE?**
    
    ### 🎯 **La Regola d'Oro:**
    
    #### **🟢 PEARSON - Per dati "perfetti"**
    **Quando usarlo:**
    - Dati **normali** (Skewness [-0.5, 0.5] + Kurtosis ≈ 0)
    - **Pochi outlier** (< 10%)
    - Relazioni **lineari**
    
    **Cosa misura:** Correlazione lineare diretta tra due variabili
    
    #### **🟡 SPEARMAN - Per dati "imperfetti"**
    **Quando usarlo:**
    - Dati **non normali** (Skewness o Kurtosis fuori range)
    - Relazioni **monotoniche** (non necessariamente lineari)
    - Quando non sei sicuro della distribuzione
    
    **Cosa misura:** Correlazione basata sui "ranghi" (posizioni ordinate)
    
    #### **🔴 KENDALL TAU - Per dati "problematici"**
    **Quando usarlo:**
    - **Molti outlier** (> 15%)
    - **Dataset piccoli** (< 50 osservazioni)
    - Quando hai **molti valori uguali**
    
    **Cosa misura:** Correlazione basata sulla "concordanza" tra coppie di osservazioni
    
    ### 📊 **Esempio di Scelta:**
    ```
    Colonna "Stipendi":
    - Skewness = 2.3 (> 1) → 🔴 Molto distorta
    - Outlier = 25% → 🔴 Molti outlier
    → RACCOMANDAZIONE: KENDALL TAU
    
    Colonna "Età":
    - Skewness = 0.2 (tra -0.5 e 0.5) → 🟢 Simmetrica
    - Kurtosis = -0.1 (≈ 0) → 🟢 Normale
    - Outlier = 3% → 🟢 Pochi outlier
    → RACCOMANDAZIONE: PEARSON
    ```
    
    ---
    
    ## 🚨 **4. GESTIONE DEGLI OUTLIER - COSA FARE CON I VALORI "STRANI"?**
    
    ### 🔍 **Come riconoscerli:**
    **Metodo IQR (Interquartile Range):**
    - Calcola Q1 (25°percentile) e Q3 (75°percentile)
    - IQR = Q3 - Q1
    - **Outlier** = valori < Q1 - 1.5×IQR o > Q3 + 1.5×IQR
    
    ### 🤔 **Cosa fare:**
    
    #### **Se sono ERRORI (es. età = 200 anni):**
    - **RIMUOVILI** senza pietà
    
    #### **Se sono REALI ma estremi (es. CEO con stipendio altissimo):**
    - **< 10% outlier**: Tienili, usa statistiche robuste (mediana, IQR)
    - **10-15% outlier**: Valuta caso per caso
    - **> 15% outlier**: Usa correlazione di **Kendall**, considera **trasformazioni**
    
    ### 💡 **Trasformazioni utili per outlier:**
    - **Logaritmo**: per dati con forte asimmetria positiva
    - **Radice quadrata**: per dati con varianza crescente
    - **Winsorizing**: sostituisci outlier con valori limite (5° e 95° percentile)
    
    ---
    
    ## 🧹 **5. PULIZIA DEI DATI - LE REGOLE D'ORO**
    
    ### ❌ **Quando RIMUOVERE una colonna:**
    1. **Un solo valore unico** → inutile (es. colonna "Paese" = sempre "Italia")
    2. **> 50% valori mancanti** → inaffidabile
    3. **> 90% valori identici** → poco informativa
    4. **Correlazione > 0.95 con un'altra colonna** → ridondante
    
    ### 🔧 **Gestione valori mancanti:**
    - **< 5% mancanti**: Rimuovi le righe
    - **5-20% mancanti**: 
      - Numeri → media/mediana (usa regola del 30%)
      - Categorie → moda o "Sconosciuto"
    - **> 20% mancanti**: Considera di rimuovere la colonna
    
    ---
    
    ## 📈 **6. INTERPRETAZIONE DELLE CORRELAZIONI**
    
    ### 🎯 **Scale di forza:**
    - **|r| < 0.3**: Correlazione **DEBOLE** (quasi nessuna relazione)
    - **0.3 ≤ |r| < 0.7**: Correlazione **MODERATA** (relazione evidente)
    - **0.7 ≤ |r| < 0.9**: Correlazione **FORTE** (relazione chiara)
    - **|r| ≥ 0.9**: Correlazione **MOLTO FORTE** (quasi perfetta)
    
    ### ⚠️ **Attenzione alla multicollinearità:**
    - **|r| > 0.8** tra due variabili → potrebbero essere ridondanti
    - Considera di tenerne solo una o usare **PCA** (Principal Component Analysis)
    
    ### 🚫 **Ricorda: Correlazione ≠ Causazione**
    - Alta correlazione NON significa che una variabile causa l'altra
    - Potrebbero esistere **variabili confondenti** o **relazioni spurie**
    
    ---
    
    ## 🎯 **7. PROCESSO DECISIONALE STEP-BY-STEP**
    
    ### **STEP 1: Analizza ogni colonna**
    1. Calcola Skewness e Kurtosis
    2. Applica regola del 30% di Malizia
    3. Conta gli outlier
    
    ### **STEP 2: Scegli le statistiche**
    - Regola 30% → Media o Mediana per riassumere
    - Normalità → Pearson, Spearman o Kendall per correlazioni
    
    ### **STEP 3: Pulisci i dati**
    - Rimuovi colonne inutili
    - Gestisci valori mancanti
    - Decidi cosa fare con gli outlier
    
    ### **STEP 4: Analizza le relazioni**
    - Calcola correlazioni con metodo appropriato
    - Identifica multicollinearità
    - Interpreta i risultati
    
    ---
    
    ## 🎓 **ESEMPIO COMPLETO**
    
    **Scenario**: Analisi stipendi aziendali
    
    ```
    Colonna "Stipendio":
    - Media = 45.000€, Std = 25.000€
    - 25.000/45.000 = 55% > 30% → USA MEDIANA ✓
    - Skewness = 1.8 > 1 → Molto distorto ✓
    - Outlier = 12% → Moderato ✓
    → Correlazione: SPEARMAN
    
    Colonna "Età":
    - Media = 35 anni, Std = 8 anni  
    - 8/35 = 23% < 30% → USA MEDIA ✓
    - Skewness = 0.1, Kurtosis = -0.2 → Normale ✓
    - Outlier = 4% → Pochi ✓
    → Correlazione: PEARSON
    
    Risultato correlazione Età-Stipendio:
    - Usa SPEARMAN (il più conservativo tra i due)
    - r = 0.65 → Correlazione MODERATA-FORTE
    - Interpretazione: tendenzialmente, persone più anziane 
      hanno stipendi più alti, ma con diverse eccezioni
    ```
    
    ---
    
    ## 💡 **SUGGERIMENTI FINALI**
    
    1. **Inizia sempre** con statistiche descrittive di base
    2. **Visualizza** i dati (grafici, istogrammi) prima di applicare test
    3. **Documenta** le scelte fatte e perché
    4. **Non fidarti** ciecamente dei numeri - verifica sempre la logica
    5. **Quando in dubbio**, scegli il metodo più conservativo
    6. **Ricorda**: È meglio essere approssimativamente giusti che precisamente sbagliati
    
    ---
    
    *🎯 Questa guida copre il 90% delle situazioni che incontrerai nell'analisi dati quotidiana*
    """)

# Upload file
uploaded_file = st.file_uploader("📁 Carica un file", type=["csv", "tsv", "xlsx", "xls", "json", "pdf", "html", "parquet", "feather"])

df = None
if uploaded_file:
    df = load_data(uploaded_file)
    if df is not None:
        st.success("✅ File caricato con successo")

        with st.expander("🧼 Opzioni di pulizia"):
            remove_dups = st.checkbox("Rimuovi duplicati", value=True)
            missing_opt = st.radio("Gestione valori mancanti", ["Mantieni", "Rimuovi", "Riempi con 0"])

        if remove_dups:
            df.drop_duplicates(inplace=True)
        if missing_opt == "Rimuovi":
            df.dropna(inplace=True)
        elif missing_opt == "Riempi con 0":
            df.fillna(0, inplace=True)

        df = clean_data(df)
        st.session_state["df_clean"] = df

        st.dataframe(df.head(), use_container_width=True)

        # --- REGOLA DEL 30% DI MALIZIA ---
        st.markdown("### 📏 Regola del 30% (Prof. Malizia)")
        st.info("**Regola**: Se la deviazione standard è < 30% della media → la media è affidabile, altrimenti usa la mediana")
        
        malizia_analysis = malizia_30_percent_rule(df)
        if malizia_analysis:
            malizia_df = pd.DataFrame(malizia_analysis).T
            st.dataframe(malizia_df.style.apply(
                lambda x: ['background-color: lightgreen' if v else 'background-color: lightcoral' 
                          for v in x] if x.name == 'mean_reliable' else [''] * len(x), axis=0
            ), use_container_width=True)

        # --- TEST DI NORMALITÀ (FISCHER) ---
        with st.expander("📊 Test di Normalità e Asimmetria (Fischer)", expanded=False):
            st.info("**Obiettivo**: Capire se i dati seguono una distribuzione normale per scegliere i test statistici giusti")
            
            # LEGENDA COMPLETA (senza expander annidato)
            st.markdown("---")
            st.markdown("### 📚 **LEGENDA: Cosa significano Skewness e Kurtosis?**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                ### 📐 **SKEWNESS (Asimmetria)**
                *Misura quanto la distribuzione è "sbilanciata" rispetto al centro*
                
                **🔢 Valori di riferimento:**
                - **`0`** = Perfettamente simmetrica (come campana)
                - **`> 0`** = Coda lunga a DESTRA (valori alti)
                - **`< 0`** = Coda lunga a SINISTRA (valori bassi)
                
                **🎯 Scale di giudizio:**
                - **`[-0.5, 0.5]`** = 🟢 **Simmetrica** (quasi normale)
                - **`[-1, -0.5]` o `[0.5, 1]`** = 🟡 **Moderatamente distorta**
                - **`< -1` o `> 1`** = 🔴 **Molto distorta** (problemática)
                
                **💡 Esempio pratico:**
                - Stipendi: spesso Skewness > 0 (pochi stipendi molto alti)
                - Voti esami: spesso Skewness < 0 (pochi voti molto bassi)
                """)
            
            with col2:
                st.markdown("""
                ### 📊 **KURTOSIS (Curtosi)**
                *Misura quanto la distribuzione è "appuntita" o "piatta"*
                
                **🔢 Regola di Fischer:**
                - **`0`** = Distribuzione **NORMALE** (perfetta)
                - **`> 0`** = Più **APPUNTITA** della normale (leptocurtica)
                - **`< 0`** = Più **PIATTA** della normale (platicurtica)
                
                **🎯 Scale di giudizio:**
                - **`|kurtosis| < 0.5`** = 🟢 **Quasi normale**
                - **`|kurtosis| < 1`** = 🟡 **Accettabile**
                - **`|kurtosis| ≥ 1`** = 🔴 **Non normale**
                
                **💡 Cosa significa:**
                - **Kurtosis alta (+)**: Molti valori vicini alla media + outlier estremi
                - **Kurtosis bassa (-)**: Valori distribuiti uniformemente
                """)
            
            st.markdown("""
            ---
            ### 🤔 **QUANDO I DATI SONO "NORMALI"?**
            
            Una distribuzione è considerata **normale** quando:
            1. **Skewness** è tra `-0.5` e `0.5` (simmetrica)
            2. **Kurtosis** è vicina a `0` (secondo Fischer)
            
            **🎯 Perché è importante?**
            - **Dati normali** → Usa correlazione di **Pearson**
            - **Dati non normali** → Usa correlazione di **Spearman**
            - **Molti outlier** → Usa correlazione di **Kendall**
            """)
            
            st.markdown("---")
            
            # Valori di riferimento rapidi
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("""
                **📐 Skewness:** `[-0.5, 0.5]` = 🟢 | `[-1, 1]` = 🟡 | `Altri` = 🔴
                """)
            with col2:
                st.markdown("""
                **📊 Kurtosis:** `≈ 0` = 🟢 Normale | `|k| < 1` = 🟡 | `|k| ≥ 1` = 🔴
                """)
            
            normality_results = normality_analysis(df)
            if normality_results:
                # Crea una tabella riassuntiva
                summary_data = []
                for col, result in normality_results.items():
                    summary_data.append({
                        'Colonna': col,
                        'Skewness': result['skewness'],
                        'Interpretazione Skewness': f"{result['skew_color']} {result['skew_classification']}",
                        'Kurtosis': result['kurtosis'],
                        'Interpretazione Kurtosis': f"{result['kurt_color']} {result['kurt_classification']}",
                        'È Normale?': "🟢 Sì" if result['is_normal'] else "🔴 No"
                    })
                
                summary_df = pd.DataFrame(summary_data)
                st.dataframe(summary_df, use_container_width=True)
                
                # Conteggio rapido
                normal_count = sum(1 for r in normality_results.values() if r['is_normal'])
                total_count = len(normality_results)
                
                if normal_count == 0:
                    st.warning(f"⚠️ Nessuna colonna ha distribuzione normale → Preferisci **Spearman** o **Kendall** per le correlazioni")
                elif normal_count == total_count:
                    st.success(f"✅ Tutte le {total_count} colonne hanno distribuzione normale → Puoi usare **Pearson** per le correlazioni")
                else:
                    st.info(f"📊 {normal_count}/{total_count} colonne hanno distribuzione normale → Valuta caso per caso")
            else:
                st.warning("Nessun dato numerico disponibile per il test di normalità")

        st.markdown("### 📌 Statistiche Numeriche Avanzate")
        num_stats = describe_numeric_advanced(df.select_dtypes(include=np.number))
        st.dataframe(num_stats)

        st.markdown("### 🚨 Outlier Rilevati")
        outlier_info = detect_outliers(df)
        for col, info in outlier_info.items():
            if info['count'] > 0:
                st.warning(f"Colonna `{col}`: {info['count']} outlier ({info['percentage']}%) [Range: {info['bounds'][0]} - {info['bounds'][1]}]")
            else:
                st.info(f"Colonna `{col}`: Nessun outlier significativo rilevato")

# --- CORRELAZIONE AVANZATA CON SUGGERIMENTI AUTOMATICI ---
st.markdown("## 🔗 Analisi di Correlazione Avanzata")

df = st.session_state.get("df_clean")
if df is not None:
    num_cols = df.select_dtypes(include=np.number)

    if not num_cols.empty:
        # Suggerimenti automatici per metodo di correlazione
        st.markdown("### 🎯 Suggerimenti per Metodo di Correlazione")
        outlier_info = detect_outliers(df)
        correlation_suggestions = suggest_correlation_method(df, outlier_info)
        
        if correlation_suggestions:
            st.info("**Raccomandazioni basate sui dati:**")
            for col, suggestion in correlation_suggestions.items():
                st.write(f"{suggestion['color']} **{col}**: {suggestion['method']} ({suggestion['reason']})")
        
        # Selezione metodo con suggerimento
        st.markdown("### 🔧 Scegli il Metodo")
        method = st.selectbox("Metodo di correlazione", ["pearson", "spearman", "kendall"])
        
        # Informazioni sui metodi
        method_info = {
            "pearson": "🟢 Ideale per dati normali e relazioni lineari",
            "spearman": "🟡 Migliore per dati non normali o relazioni monotoniche",
            "kendall": "🔴 Robusto con molti outlier o dataset piccoli"
        }
        st.info(method_info[method])
        
        corr = num_cols.corr(method=method)
        st.dataframe(corr.style.background_gradient(cmap="coolwarm"), use_container_width=True)

        if len(num_cols.columns) >= 2:
            fig = px.imshow(
                corr,
                text_auto=True,
                title=f"Matrice di Correlazione ({method.title()})",
                color_continuous_scale='RdBu_r',
                zmin=-1, zmax=1
            )
            st.plotly_chart(fig, use_container_width=True)

        high_corr = []
        for i in range(len(corr.columns)):
            for j in range(i+1, len(corr.columns)):
                val = corr.iloc[i, j]
                if abs(val) > 0.8:
                    high_corr.append((corr.index[i], corr.columns[j], val))

        if high_corr:
            st.warning("⚠️ Correlazioni elevate rilevate:")
            for a, b, v in high_corr:
                st.write(f"🔸 **{a}** e **{b}** → coeff = {v:.2f} → potenziale ridondanza")

        # Suggerimenti correlazione
        with st.expander("📘 Guida all'Interpretazione delle Correlazioni"):
            st.markdown("""
**Correlazione positiva**: all'aumentare di una variabile, anche l'altra tende ad aumentare.

**Correlazione negativa**: all'aumentare di una variabile, l'altra tende a diminuire.

**Vicino a 0**: nessuna relazione lineare evidente.

**Regole del Prof. Malizia per scegliere il metodo:**
- **Dati normali** (Kurtosis ≈ 0, Asimmetria -0.5/0.5) → **Pearson**
- **Dati non normali** → **Spearman**
- **Molti outlier** (>15%) → **Kendall Tau**

**Attenzione a:**
- Correlazioni > 0.8 → multicollinearità
- Valori inaspettati → verifica anomalie o variabili confondenti
- Usa **PCA** per ridurre la dimensionalità se trovi molte variabili correlate.
            """)
    else:
        st.info("Nessuna colonna numerica disponibile per la correlazione.")
else:
    st.warning("⚠️ Carica prima un dataset valido.")

# --- CONSIGLI AUTOMATICI POTENZIATI ---
st.markdown("## 💡 Suggerimenti Automatici (Smart Advisor)")

if df is not None:
    messages = []
    
    # Consigli sulla dimensione del dataset
    if df.shape[0] < 50:
        messages.append("📉 Pochi dati: i risultati potrebbero non essere rappresentativi.")
    
    # Consigli sulle colonne
    for col in df.columns:
        if df[col].nunique() == 1:
            messages.append(f"🟨 La colonna `{col}` ha un solo valore unico → poco informativa.")
        elif df[col].nunique() / df.shape[0] > 0.9:
            messages.append(f"🟨 La colonna `{col}` ha altissima cardinalità ({df[col].nunique()} valori unici).")
        elif df[col].dtype == 'float' and df[col].std() < 1e-3:
            messages.append(f"🔍 La colonna `{col}` ha una varianza molto bassa → quasi costante.")

    # Consigli basati sulla regola del 30% di Malizia
    malizia_results = malizia_30_percent_rule(df)
    for col, result in malizia_results.items():
        if not result['mean_reliable']:
            messages.append(f"📏 **Regola Malizia**: Per `{col}` usa la **mediana** ({result['median']}) invece della media (std = {result['std_percent']}%)")

    # Consigli basati sulla normalità
    normality_results = normality_analysis(df)
    for col, result in normality_results.items():
        if not result['is_normal']:
            if result['skew_classification'] == "Molto distorti":
                messages.append(f"📊 `{col}` è molto distorta (asimmetria = {result['skewness']}) → considera trasformazioni (log, sqrt)")
            if result['kurt_classification'] == "Non normale":
                messages.append(f"📊 `{col}` non segue distribuzione normale (kurtosis = {result['kurtosis']}) → usa test non parametrici")

    # Consigli sugli outlier
    for col, out in detect_outliers(df).items():
        if out["percentage"] > 10:
            messages.append(f"🚨 `{col}` ha {out['percentage']}% outlier → potrebbe influenzare media o regressioni.")

    if messages:
        for msg in messages:
            st.info(msg)
    else:
        st.success("✅ Nessun problema evidente rilevato. Dati appaiono bilanciati.")

# --- LEGENDA INTERATTIVA POTENZIATA ---
with st.expander("📘 Guida Completa alle Analisi (Regole Prof. Malizia)"):
    st.markdown("""
### 📏 **Regola del 30% (Prof. Malizia)**
- Se `std/media < 30%` → **Media affidabile**
- Se `std/media ≥ 30%` → **Usa la Mediana** (più robusta)

### 📊 **Test di Normalità (Fischer)**
- **Kurtosis ≈ 0**: Distribuzione normale
- **Asimmetria**:
  - `-0.5 ≤ asimmetria ≤ 0.5`: Dati simmetrici ✅
  - `-1 ≤ asimmetria < -0.5` o `0.5 < asimmetria ≤ 1`: Moderatamente distorti ⚠️
  - `asimmetria < -1` o `asimmetria > 1`: Molto distorti ❌

### 🔗 **Scelta Metodo di Correlazione**
- **Dati normali** → **Pearson**
- **Dati non normali** → **Spearman**
- **Molti outlier** → **Kendall Tau**

### 🔎 **Quando rimuovere una colonna?**
- Ha **1 solo valore** → sempre
- Ha **>90% valori uguali** → probabilmente inutile
- Ha **correlazione > 0.9 con un'altra** → valuta di tenerne solo una
- **Regola Malizia**: std > 30% media → considera mediana invece di media

### 📈 **Gestione Outlier**
- Se sono **errori** → rimuovili
- Se sono **reali** → usa statistiche robuste (mediana, IQR)
- **>15% outlier** → usa correlazione di Kendall
    """)



# Enhanced pyNarrative section for your Streamlit app
# Replace the existing pyNarrative section with this improved version

# === SEZIONE PYNARRATIVE POTENZIATA ===
st.markdown("## 📖 Generazione Narrativa con pyNarrative")

if df is not None:
    st.info("💡 **Come utilizzare il DataFrame**: Il tuo dataset pulito è disponibile come variabile `df`")
    
    # Quick data summary for user reference
    with st.expander("📊 Riassunto del tuo Dataset", expanded=False):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Righe", f"{df.shape[0]:,}")
        with col2:
            st.metric("Colonne", df.shape[1])
        with col3:
            st.metric("Numeriche", len(df.select_dtypes(include=np.number).columns))
        with col4:
            st.metric("Categoriche", len(df.select_dtypes(include='object').columns))
        
        # Show column details
        st.markdown("**📋 Colonne disponibili:**")
        cols_data = []
        for col in df.columns:
            dtype = str(df[col].dtype)
            null_count = df[col].isnull().sum()
            unique_count = df[col].nunique()
            cols_data.append({
                'Colonna': col,
                'Tipo': dtype,
                'Valori Unici': f"{unique_count:,}",
                'Valori Nulli': null_count,
                'Esempio': str(df[col].iloc[0]) if len(df) > 0 else "N/A"
            })
        
        cols_df = pd.DataFrame(cols_data)
        st.dataframe(cols_df, use_container_width=True)
    
    with st.expander("📚 Guida pyNarrative Completa", expanded=False):
        st.markdown("""
        ### 🎯 **Template di Base pyNarrative**
        
        **Il tuo DataFrame pulito è già disponibile come variabile `df`**
        
        #### 🟢 **Template 1: Grafico a Barre Semplice**
        ```python
        import pyNarrative as pn
        import altair as alt
        
        storia = pn.Story(df, font='Verdana')
        grafico = (storia
            .mark_bar()
            .encode(
                x='nome_colonna_x:Q',  # :Q per quantitativo, :N per nominale
                y='nome_colonna_y:Q'
            )
            .add_title(title='Il Mio Grafico')
        )
        visualizzazione = grafico.render()
        st.altair_chart(visualizzazione, use_container_width=True)
        ```
        
        #### 🟡 **Template 2: Scatter Plot con Colori**
        ```python
        import pyNarrative as pn
        import altair as alt
        
        storia = pn.Story(df, font='Verdana')
        grafico = (storia
            .mark_circle(size=60)
            .encode(
                x='nome_colonna_x:Q',
                y='nome_colonna_y:Q',
                color='nome_colonna_colore:Q',
                tooltip=['nome_colonna_x:Q', 'nome_colonna_y:Q']
            )
            .add_title(title='Correlazione tra Variabili', title_font_size=18)
            .add_context(
                text=("Questo grafico mostra la relazione",
                      "tra le due variabili principali"),
                position="right"
            )
        )
        visualizzazione = grafico.render()
        st.altair_chart(visualizzazione, use_container_width=True)
        ```
        
        #### 🔴 **Template 3: Grafico Completo con Annotazioni**
        ```python
        import pyNarrative as pn
        import altair as alt
        
        storia = pn.Story(df, font='Verdana')
        grafico = (storia
            .mark_bar(size=15)
            .encode(
                x=alt.X('nome_colonna_x:Q', axis=alt.Axis(format='d')),
                y=alt.Y('nome_colonna_y:Q', title='Titolo Asse Y'),
                color=alt.Color('nome_colonna_colore:Q', scale=alt.Scale(scheme='viridis'))
            )
            .add_title(
                title='Analisi Dettagliata',
                title_font_size=22,
                dy=30
            )
            .add_context(
                text=("Analisi basata sui dati",
                      "del Prof. Malizia"),
                position="right",
                dx=190,
                dy=-30
            )
            .add_next_steps(
                mode='stair_steps',
                texts=[
                    "Verifica outlier",
                    ["Applica regola", "del 30%"],
                    ["Calcola correlazioni", "appropriate"]
                ],
                title="Prossimi Passi",
                title_font_size=16
            )
            .add_source(
                text=f"Fonte: Dataset con {len(df):,} righe",
                position="bottom",
                font_size=12
            )
        )
        visualizzazione = grafico.render().configure_axis(grid=False)
        st.altair_chart(visualizzazione, use_container_width=True)
        ```
        
        ### 🔧 **Tipi di Encoding Altair:**
        - **`:Q`** = Quantitativo (numeri continui)
        - **`:O`** = Ordinale (categorie ordinate)
        - **`:N`** = Nominale (categorie senza ordine)
        - **`:T`** = Temporale (date/orari)
        
        ### 🎨 **Schemi di Colori Disponibili:**
        `'viridis'`, `'plasma'`, `'blues'`, `'reds'`, `'greens'`, `'purples'`, `'oranges'`
        """)
    
    # Template selector for easier use
    st.markdown("### 🚀 Generatore di Template")
    
    template_choice = st.selectbox(
        "Scegli un template di partenza:",
        [
            "Template Personalizzato",
            "Grafico a Barre Semplice",
            "Scatter Plot con Colori", 
            "Analisi Temporale",
            "Confronto Categorico",
            "Distribuzione con Istogramma"
        ]
    )
    
    # Get numeric and categorical columns
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    
    if template_choice != "Template Personalizzato":
        st.markdown("#### 🎛️ Configura il Template")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if numeric_cols:
                x_col = st.selectbox("Colonna X:", [""] + numeric_cols + categorical_cols + datetime_cols)
            else:
                x_col = st.selectbox("Colonna X:", [""] + categorical_cols + datetime_cols)
        
        with col2:
            if numeric_cols:
                y_col = st.selectbox("Colonna Y:", [""] + numeric_cols)
            else:
                y_col = ""
                st.warning("Nessuna colonna numerica disponibile per Y")
        
        color_col = st.selectbox("Colonna Colore (opzionale):", ["Nessuna"] + numeric_cols + categorical_cols)
        
        chart_title = st.text_input("Titolo del grafico:", value=f"Analisi di {x_col} vs {y_col}" if x_col and y_col else "Il Mio Grafico")
        
        # Generate template based on selection
        if x_col and y_col:
            if template_choice == "Grafico a Barre Semplice":
                generated_code = f"""import pyNarrative as pn
import altair as alt

storia = pn.Story(df, font='Verdana')
grafico = (storia
    .mark_bar()
    .encode(
        x='{x_col}:{"Q" if x_col in numeric_cols else "N"}',
        y='{y_col}:Q'{"," if color_col != "Nessuna" else ""}
        {"color='" + color_col + ":Q'" if color_col != "Nessuna" and color_col in numeric_cols else ""}
        {"color='" + color_col + ":N'" if color_col != "Nessuna" and color_col in categorical_cols else ""}
    )
    .add_title(title='{chart_title}')
)
visualizzazione = grafico.render()
st.altair_chart(visualizzazione, use_container_width=True)"""
            
            elif template_choice == "Scatter Plot con Colori":
                generated_code = f"""import pyNarrative as pn
import altair as alt

storia = pn.Story(df, font='Verdana')
grafico = (storia
    .mark_circle(size=60)
    .encode(
        x='{x_col}:Q',
        y='{y_col}:Q'{"," if color_col != "Nessuna" else ""}
        {"color='" + color_col + ":Q'," if color_col != "Nessuna" and color_col in numeric_cols else ""}
        {"color='" + color_col + ":N'," if color_col != "Nessuna" and color_col in categorical_cols else ""}
        tooltip=['{x_col}:Q', '{y_col}:Q']
    )
    .add_title(title='{chart_title}', title_font_size=18)
    .add_context(
        text=("Correlazione tra {x_col}",
              "e {y_col}"),
        position="right"
    )
)
visualizzazione = grafico.render()
st.altair_chart(visualizzazione, use_container_width=True)"""
            
            elif template_choice == "Analisi Temporale":
                generated_code = f"""import pyNarrative as pn
import altair as alt

storia = pn.Story(df, font='Verdana')
grafico = (storia
    .mark_line(point=True)
    .encode(
        x='{x_col}:{"T" if x_col in datetime_cols else "Q"}',
        y='{y_col}:Q'{"," if color_col != "Nessuna" else ""}
        {"color='" + color_col + ":N'" if color_col != "Nessuna" else ""}
    )
    .add_title(
        title='{chart_title}',
        title_font_size=20
    )
    .add_context(
        text=("Evoluzione temporale", "dei dati nel tempo"),
        position="right"
    )
    .add_source(
        text="Fonte: Dataset analizzato",
        position="bottom"
    )
)
visualizzazione = grafico.render()
st.altair_chart(visualizzazione, use_container_width=True)"""
            
            elif template_choice == "Confronto Categorico":
                generated_code = f"""import pyNarrative as pn
import altair as alt

storia = pn.Story(df, font='Verdana')
grafico = (storia
    .mark_bar()
    .encode(
        x='{x_col}:N' if x_col in categorical_cols else f'{x_col}:Q',
        y='{y_col}:Q',
        color=alt.Color('{x_col}:N' if x_col in categorical_cols else f'{x_col}:Q', 
                       scale=alt.Scale(scheme='category10'))
    )
    .add_title(title='{chart_title}', title_font_size=18)
    .add_next_steps(
        mode='bullet_points',
        texts=[
            "Identifica categorie dominanti",
            "Analizza differenze significative",
            "Verifica distribuzioni"
        ],
        title="Insights"
    )
)
visualizzazione = grafico.render()
st.altair_chart(visualizzazione, use_container_width=True)"""
            
            elif template_choice == "Distribuzione con Istogramma":
                generated_code = f"""import pyNarrative as pn
import altair as alt

storia = pn.Story(df, font='Verdana')
grafico = (storia
    .mark_bar()
    .encode(
        x=alt.X('{x_col}:Q', bin=alt.Bin(maxbins=20)),
        y='count():Q'
    )
    .add_title(title='Distribuzione di {x_col}', title_font_size=18)
    .add_context(
        text=("Analisi della distribuzione",
              "secondo regole Malizia"),
        position="right"
    )
)
visualizzazione = grafico.render()
st.altair_chart(visualizzazione, use_container_width=True)"""
        else:
            generated_code = "# Seleziona le colonne X e Y per generare il template"
    else:
        # Default template for custom
        generated_code = """import pyNarrative as pn
import altair as alt

# Il DataFrame pulito è disponibile come 'df'
storia = pn.Story(df, font='Verdana')

# Esempio base - sostituisci con le tue colonne
grafico = (storia
    .mark_circle(size=60)
    .encode(
        x='nome_colonna_x:Q',  # Sostituisci con la tua colonna
        y='nome_colonna_y:Q'   # Sostituisci con la tua colonna
    )
    .add_title(title='Il Mio Grafico Personalizzato')
)

visualizzazione = grafico.render()
st.altair_chart(visualizzazione, use_container_width=True)"""
    
    # Code editor
    st.markdown("### 💻 Editor Codice pyNarrative")
    
    user_code = st.text_area(
        "Codice pyNarrative:",
        value=generated_code,
        height=400,
        help="Modifica il codice secondo le tue esigenze. Usa st.altair_chart() per visualizzare i grafici."
    )
    
    # Execute button
    col1, col2, col3 = st.columns([1, 2, 2])
    
    with col1:
        execute_button = st.button("🚀 Esegui", type="primary")
    
    with col2:
        if st.button("📋 Copia Template"):
            st.code(user_code, language="python")
    
    with col3:
        clear_button = st.button("🗑️ Pulisci Output")
    
    # Execution area
    if execute_button:
        if user_code.strip():
            st.markdown("### 📊 Risultato:")
            
            try:
                # Create execution environment
                exec_globals = {
                    'df': df,
                    'pd': pd,
                    'np': np,
                    'st': st,
                    'alt': None,  # Will be imported by user code
                    'pn': None,   # Will be imported by user code
                    '__builtins__': __builtins__
                }
                
                # Execute the code
                exec(user_code, exec_globals)
                
                st.success("✅ Codice eseguito con successo!")
                    
            except ImportError as e:
                if "pyNarrative" in str(e):
                    st.error("❌ pyNarrative non trovato!")
                    with st.expander("💡 Come installare pyNarrative"):
                        st.markdown("""
                        **Installazione pyNarrative:**
                        ```bash
                        pip install pyNarrative
                        ```
                        
                        **Se sviluppato localmente:**
                        ```bash
                        pip install -e /path/to/pyNarrative
                        ```
                        
                        **Verifica installazione:**
                        ```python
                        import pyNarrative as pn
                        print(pn.__version__)
                        ```
                        """)
                elif "altair" in str(e):
                    st.error("❌ Altair non trovato!")
                    st.code("pip install altair", language="bash")
                else:
                    st.error(f"❌ Errore di importazione: {str(e)}")
                    
            except Exception as e:
                st.error(f"❌ Errore durante l'esecuzione:")
                st.code(str(e), language="python")
                
                with st.expander("🔧 Suggerimenti per Debug"):
                    st.markdown("""
                    **Controlli comuni:**
                    1. ✅ Verifica che le colonne specificate esistano nel DataFrame
                    2. ✅ Controlla la sintassi del codice Python
                    3. ✅ Assicurati che pyNarrative e Altair siano installati
                    4. ✅ Usa `st.altair_chart()` per mostrare grafici in Streamlit
                    5. ✅ Verifica i tipi di encoding (:Q, :N, :O, :T)
                    
                    **Colonne disponibili nel tuo DataFrame:**
                    """)
                    st.write(list(df.columns))
        else:
            st.warning("⚠️ Inserisci del codice prima di eseguire!")
    
    # Clear output section
    if clear_button:
        st.rerun()
    
    # Examples gallery
    with st.expander("🎨 Galleria di Esempi Avanzati", expanded=False):
        st.markdown("""
        ### 🔥 Esempi Professionali
        
        #### 📊 **Dashboard Completo**
        ```python
        import pyNarrative as pn
        import altair as alt
        
        storia = pn.Story(df, font='Arial')
        
        # Grafico principale con tutte le features
        grafico = (storia
            .mark_bar(size=20)
            .encode(
                x=alt.X('categoria:N', sort='-y', axis=alt.Axis(labelAngle=-45)),
                y=alt.Y('valore:Q', title='Valori Misurati'),
                color=alt.Color('media:Q', 
                               scale=alt.Scale(scheme='viridis'),
                               legend=alt.Legend(title="Media Mobile"))
            )
            .add_title(
                title='Dashboard Analisi Completa',
                title_font_size=24,
                dy=40
            )
            .add_context(
                text=("Analisi basata sulla regola del 30%",
                      "del Prof. Malizia per la valutazione",
                      "dell'affidabilità delle medie"),
                position="right",
                dx=200,
                dy=-50,
                font_size=12
            )
            .add_next_steps(
                mode='stair_steps',
                texts=[
                    "Verifica normalità dati",
                    ["Applica test di", "correlazione appropriato"],
                    ["Identifica e gestisci", "outlier significativi"],
                    "Genera report finale"
                ],
                title="Piano di Analisi",
                title_font_size=18,
                dx=200,
                dy=50
            )
            .add_annotation(
                x_point=2,
                y_point=150,
                annotation_text="Picco anomalo",
                label_size=12,
                arrow_direction='left'
            )
            .add_line(
                value=100,
                orientation='horizontal',
                color='red',
                stroke_width=2,
                stroke_dash=[5, 5]
            )
            .add_source(
                text=f"Fonte: Dataset con {len(df):,} osservazioni - Elaborazione {pd.Timestamp.now().strftime('%Y-%m-%d')}",
                position="bottom",
                font_size=10
            )
        )
        
        # Rendering con configurazioni avanzate
        visualizzazione = (grafico
            .render()
            .configure_axis(
                grid=False,
                domain=False
            )
            .configure_view(
                strokeWidth=0
            )
            .resolve_scale(
                color='independent'
            )
        )
        
        st.altair_chart(visualizzazione, use_container_width=True)
        ```
        
        #### 🎯 **Analisi Multi-Panel**
        ```python
        import pyNarrative as pn
        import altair as alt
        
        storia = pn.Story(df, font='Verdana')
        
        # Crea visualizzazione a pannelli multipli
        base = storia.mark_circle(size=50).encode(
            color=alt.Color('categoria:N', scale=alt.Scale(scheme='category10'))
        )
        
        panel1 = base.encode(x='x1:Q', y='y1:Q').properties(title='Vista 1')
        panel2 = base.encode(x='x2:Q', y='y2:Q').properties(title='Vista 2')
        
        combined = alt.hconcat(panel1, panel2).resolve_scale(color='shared')
        
        st.altair_chart(combined, use_container_width=True)
        ```
        """)
    
    # Save functionality
    st.markdown("### 💾 Salvataggio")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📥 Scarica Codice"):
            st.download_button(
                label="💾 Download .py",
                data=user_code,
                file_name=f"pynarrative_analysis_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.py",
                mime="text/plain"
            )
    
    with col2:
        if st.button("📊 Salva Report"):
            report_content = f"""# Report Analisi pyNarrative
# Generato: {pd.Timestamp.now()}
# Dataset: {df.shape[0]} righe, {df.shape[1]} colonne

# Statistiche Dataset:
# - Colonne numeriche: {len(numeric_cols)}
# - Colonne categoriche: {len(categorical_cols)}
# - Colonne datetime: {len(datetime_cols)}

# Codice utilizzato:
{user_code}

# Colonne disponibili:
{list(df.columns)}
"""
            st.download_button(
                label="📋 Download Report",
                data=report_content,
                file_name=f"report_analisi_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.md",
                mime="text/markdown"
            )

else:
    st.warning("⚠️ Carica prima un dataset per utilizzare pyNarrative!")
    
    # Show example even without data
    with st.expander("👀 Anteprima pyNarrative (senza dati)", expanded=False):
        st.markdown("""
        ### 🎯 Esempio di utilizzo pyNarrative:
        
        ```python
        import pyNarrative as pn
        import altair as alt
        import pandas as pd
        
        # Carica i tuoi dati
        df = pd.read_csv('tuoi_dati.csv')
        
        # Crea la storia
        storia = pn.Story(df, font='Verdana')
        
        # Costruisci il grafico
        grafico = (storia
            .mark_bar()
            .encode(x='colonna_x:Q', y='colonna_y:Q')
            .add_title(title='Il Mio Grafico')
            .add_context(text=("Descrizione", "del grafico"))
        )
        
        # Visualizza in Streamlit
        visualizzazione = grafico.render()
        st.altair_chart(visualizzazione, use_container_width=True)
        ```
        """)