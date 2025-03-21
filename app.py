import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from scipy import stats
import statsmodels.api as sm
from datetime import datetime
import joblib
import scipy.stats as stats
import plotly.subplots as sp
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score

# Configurações globais
plt.style.use('ggplot')

def reduce_csv_size(input_csv, output_parquet, usecols=None, chunksize=10**6, encoding='latin1', sep=';'):
    """
    Lê o CSV pesado em chunks, realiza downcasting dos tipos numéricos 
    e salva o DataFrame em formato Parquet com compressão Snappy.
    
    Parameters:
    -----------
    input_csv : str
        Caminho para o arquivo CSV de entrada
    output_parquet : str
        Caminho para o arquivo Parquet de saída
    usecols : list, optional
        Lista de colunas para serem carregadas
    chunksize : int, optional
        Tamanho do chunk para processamento
    encoding : str, optional
        Codificação do arquivo CSV (default: 'latin1')
    sep : str, optional
        Separador usado no arquivo CSV (default: ';')
    """
    # Estimar o número total de linhas para controle de progresso
    try:
        # Primeira tentativa - método rápido para arquivos pequenos
        with open(input_csv, 'r', encoding=encoding) as f:
            total_lines = sum(1 for _ in f)
    except:
        # Para arquivos muito grandes que causariam problemas de memória
        # usamos uma estimativa baseada no tamanho do arquivo
        file_size = os.path.getsize(input_csv)
        # Analisamos as primeiras 1000 linhas para ter uma média
        with open(input_csv, 'r', encoding=encoding) as f:
            sample_lines = 1000
            sample_text = ''.join(f.readline() for _ in range(sample_lines))
        avg_line_size = len(sample_text) / sample_lines
        total_lines = int(file_size / avg_line_size)
    
    estimated_chunks = int(total_lines / chunksize) + 1
    
    chunks = []
    processed_chunks = 0
    
    # Imprimir informações para debug
    print(f"Processando arquivo: {input_csv}")
    print(f"Tamanho estimado: {total_lines:,} linhas")
    print(f"Chunks estimados: {estimated_chunks}")
    
    # Processar o arquivo em chunks
    for chunk in pd.read_csv(input_csv, usecols=usecols, chunksize=chunksize, 
                           encoding=encoding, sep=sep, low_memory=False):
        # Downcasting de colunas numéricas inteiras
        for col in chunk.select_dtypes(include=['int64']).columns:
            chunk[col] = pd.to_numeric(chunk[col], downcast='integer')
        # Downcasting de colunas numéricas float
        for col in chunk.select_dtypes(include=['float64']).columns:
            chunk[col] = pd.to_numeric(chunk[col], downcast='float')
        chunks.append(chunk)
        
        # Atualizar progresso
        processed_chunks += 1
        if processed_chunks % 10 == 0 or processed_chunks == estimated_chunks:
            percent = min(100, int((processed_chunks / estimated_chunks) * 100))
            print(f"Progresso: {percent}% ({processed_chunks}/{estimated_chunks} chunks)")
    
    print("Concatenando chunks...")
    df_reduced = pd.concat(chunks, ignore_index=True)
    
    print(f"Salvando arquivo Parquet: {output_parquet}")
    df_reduced.to_parquet(output_parquet, compression='snappy', index=False)
    
    print("Processo concluído!")
    return df_reduced

@st.cache_data
def load_data(parquet_file):
    """Carrega os dados a partir do arquivo Parquet utilizando cache."""
    df = pd.read_parquet(parquet_file)
    return df

def detect_anomalies_iqr(df, column='QT_VOTOS'):
    """
    Marca como anomalia valores fora de [Q1 - 1.5*IQR, Q3 + 1.5*IQR].
    Retorna o DataFrame com uma coluna 'ANOMALIA_IQR' booleana.
    """
    df = df.copy()
    # Remover NaN
    df = df[df[column].notna()]
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lim_inferior = Q1 - 1.5 * IQR
    lim_superior = Q3 + 1.5 * IQR
    
    df['ANOMALIA_IQR'] = ~df[column].between(lim_inferior, lim_superior)
    return df

def detect_anomalies_zscore(df, column='QT_VOTOS', z_thresh=3.0):
    """
    Marca como anomalia valores cujo z-score (desvio em relação à média)
    é maior que z_thresh (padrão=3).
    Retorna o DataFrame com a coluna 'ANOMALIA_ZSCORE' booleana.
    """
    df = df.copy()
    df = df[df[column].notna()]
    
    mean_val = df[column].mean()
    std_val = df[column].std()
    # Evita divisão por zero
    if std_val == 0:
        df['ANOMALIA_ZSCORE'] = False
        return df

    df['Z_SCORE'] = (df[column] - mean_val) / std_val
    df['ANOMALIA_ZSCORE'] = df['Z_SCORE'].abs() > z_thresh
    return df

def detect_anomalies_isolation_forest(df, columns=['QT_VOTOS']):
    """
    Aplica Isolation Forest nas colunas especificadas. 
    Retorna o DataFrame com a coluna 'ANOMALIA_IF' booleana.
    """
    df = df.copy()
    # Remover linhas com NaN nessas colunas
    df = df.dropna(subset=columns)
    
    # Treina o Isolation Forest
    iso = IsolationForest(n_estimators=100, contamination='auto', random_state=42)
    iso.fit(df[columns])
    
    # predict() retorna 1 para normal e -1 para anomalia
    df['IF_PRED'] = iso.predict(df[columns])
    df['ANOMALIA_IF'] = df['IF_PRED'] == -1
    return df

def detect_anomalies_dbscan(df, columns=['QT_VOTOS'], eps=None, min_samples=5):
    """
    Aplica DBSCAN para clustering baseado em densidade e identificação de outliers.
    Retorna o DataFrame com uma coluna 'ANOMALIA_DBSCAN' booleana.
    """
    df = df.copy()
    df_subset = df.dropna(subset=columns)
    
    # Normalização dos dados para DBSCAN
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_subset[columns])
    
    # Define eps automaticamente se não fornecido (distância média aos k vizinhos mais próximos)
    if eps is None:
        from sklearn.neighbors import NearestNeighbors
        neigh = NearestNeighbors(n_neighbors=min_samples)
        neigh.fit(X_scaled)
        distances, _ = neigh.kneighbors(X_scaled)
        distances = np.sort(distances[:, min_samples-1])
        eps = np.percentile(distances, 90)  # 90° percentil para eps
    
    # Aplicar DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = dbscan.fit_predict(X_scaled)
    
    # Em DBSCAN, clusters com valor -1 são outliers
    df_subset['ANOMALIA_DBSCAN'] = (clusters == -1)
    
    # Mesclar resultados de volta ao DataFrame original
    result = df.copy()
    result.loc[df_subset.index, 'ANOMALIA_DBSCAN'] = df_subset['ANOMALIA_DBSCAN'].values
    result['ANOMALIA_DBSCAN'] = result['ANOMALIA_DBSCAN'].fillna(False)
    
    return result

def detect_anomalies_gmm(df, columns=['QT_VOTOS'], n_components=2, threshold=0.01):
    """
    Aplica Gaussian Mixture Model para identificar outliers com base na probabilidade.
    Retorna o DataFrame com uma coluna 'ANOMALIA_GMM' booleana.
    """
    df = df.copy()
    df_subset = df.dropna(subset=columns)
    
    # Normalização dos dados para GMM
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_subset[columns])
    
    # Aplicar GMM
    gmm = GaussianMixture(n_components=n_components, random_state=42)
    gmm.fit(X_scaled)
    
    # Obter probabilidades e identificar outliers
    probs = gmm.score_samples(X_scaled)
    df_subset['GMM_SCORE'] = probs
    
    # Usar um threshold baseado no percentil para identificar anomalias
    threshold_value = np.percentile(probs, threshold * 100)  # threshold como percentil
    df_subset['ANOMALIA_GMM'] = probs < threshold_value
    
    # Mesclar resultados de volta ao DataFrame original
    result = df.copy()
    result.loc[df_subset.index, 'ANOMALIA_GMM'] = df_subset['ANOMALIA_GMM'].values
    result['ANOMALIA_GMM'] = result['ANOMALIA_GMM'].fillna(False)
    
    return result

def analyze_multivariate_anomalies(df, columns=['QT_VOTOS', 'QT_ELEITORES']):
    """
    Realiza análise multivariada utilizando PCA para identificar anomalias.
    Retorna o DataFrame com colunas de PCA e anomalias identificadas.
    """
    df = df.copy()
    df_subset = df.dropna(subset=columns)
    
    # Normalização dos dados para PCA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_subset[columns])
    
    # Aplicar PCA
    pca = PCA(n_components=min(len(columns), 2))  # Reduz para 2D para visualização
    principal_components = pca.fit_transform(X_scaled)
    
    # Adicionar componentes principais ao DataFrame
    df_subset['PC1'] = principal_components[:, 0]
    if principal_components.shape[1] > 1:
        df_subset['PC2'] = principal_components[:, 1]
    
    # Calcular distância de Mahalanobis para identificar outliers multivariados
    center = np.mean(X_scaled, axis=0)
    cov = np.cov(X_scaled.T)
    
    # Pseudo-inversa para lidar com colinearidade
    inv_cov = np.linalg.pinv(cov)
    
    # Calcular distâncias de Mahalanobis
    mahalanobis_dist = np.array(
        [np.sqrt(np.dot(np.dot((x - center).T, inv_cov), (x - center))) for x in X_scaled]
    )
    
    df_subset['MAHALANOBIS_DIST'] = mahalanobis_dist
    
    # Identificar anomalias baseadas na distribuição qui-quadrado
    # Para distribuição multivariada, os quadrados das distâncias seguem qui-quadrado com df=dimensionalidade
    dof = len(columns)
    chi2_threshold = stats.chi2.ppf(0.975, df=dof)  # 97.5% intervalo de confiança
    df_subset['ANOMALIA_MAHALANOBIS'] = df_subset['MAHALANOBIS_DIST'] > chi2_threshold
    
    # Mesclar resultados de volta ao DataFrame original
    result = df.copy()
    if 'PC1' in df_subset.columns:
        result.loc[df_subset.index, 'PC1'] = df_subset['PC1'].values
    if 'PC2' in df_subset.columns:
        result.loc[df_subset.index, 'PC2'] = df_subset['PC2'].values
    result.loc[df_subset.index, 'MAHALANOBIS_DIST'] = df_subset['MAHALANOBIS_DIST'].values
    result.loc[df_subset.index, 'ANOMALIA_MAHALANOBIS'] = df_subset['ANOMALIA_MAHALANOBIS'].values
    
    # Preencher valores ausentes com False para anomalias
    result['ANOMALIA_MAHALANOBIS'] = result['ANOMALIA_MAHALANOBIS'].fillna(False)
    
    return result

def perform_statistical_tests(df, column='QT_VOTOS', group_col='NM_MUNICIPIO'):
    """
    Realiza testes estatísticos para verificar distribuições e identifica possíveis anomalias.
    Retorna um DataFrame com os resultados dos testes.
    """
    results = []
    
    # Agrupar por coluna de agrupamento (ex: município)
    groups = df[df[column].notna()].groupby(group_col)
    
    for name, group in groups:
        if len(group) < 5:  # Ignorar grupos muito pequenos
            continue
        
        # Teste de normalidade (Shapiro-Wilk)
        if len(group) < 5000:  # Shapiro-Wilk é limitado a amostras menores
            try:
                stat_shapiro, p_shapiro = stats.shapiro(group[column])
                is_normal = p_shapiro > 0.05
            except:
                stat_shapiro, p_shapiro, is_normal = None, None, None
        else:
            stat_shapiro, p_shapiro, is_normal = None, None, None
            
        # Teste alternativo de normalidade para amostras maiores (D'Agostino)
        try:
            stat_dagostino, p_dagostino = stats.normaltest(group[column])
            is_normal_alt = p_dagostino > 0.05
        except:
            stat_dagostino, p_dagostino, is_normal_alt = None, None, None
        
        # Estatísticas descritivas
        count = len(group)
        mean = group[column].mean()
        median = group[column].median()
        std = group[column].std()
        skew = group[column].skew()
        kurtosis = group[column].kurtosis()
        
        # Calculando o Coeficiente de Variação (CV)
        cv = (std / mean) if mean != 0 else None
        
        # Outliers via IQR
        q1 = group[column].quantile(0.25)
        q3 = group[column].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers_count = ((group[column] < lower_bound) | (group[column] > upper_bound)).sum()
        outliers_pct = (outliers_count / count) * 100 if count > 0 else 0
        
        # Calculando a razão média votos/eleitores, se possível
        if 'QT_ELEITORES' in group.columns:
            razao_media = (group[column] / group['QT_ELEITORES'].replace(0, np.nan)).mean()
        else:
            razao_media = None
        
        # Armazenar resultados
        results.append({
            group_col: name,
            'CONTAGEM': count,
            'MÉDIA': mean,
            'MEDIANA': median,
            'DESVIO_PADRÃO': std,
            'CV': cv,  # Coeficiente de Variação
            'ASSIMETRIA': skew,
            'CURTOSE': kurtosis,
            'OUTLIERS': outliers_count,
            'OUTLIERS_PCT': outliers_pct,
            'SHAPIRO_STAT': stat_shapiro,
            'SHAPIRO_P': p_shapiro,
            'NORMAL_SHAPIRO': is_normal,
            'DAGOSTINO_STAT': stat_dagostino,
            'DAGOSTINO_P': p_dagostino,
            'NORMAL_DAGOSTINO': is_normal_alt,
            'RAZAO_MÉDIA': razao_media
        })
    
    # Converter para DataFrame
    results_df = pd.DataFrame(results)
    
    # Classificar resultados por percentual de outliers (decrescente)
    if 'OUTLIERS_PCT' in results_df.columns:
        results_df = results_df.sort_values('OUTLIERS_PCT', ascending=False)
    
    return results_df

def analyze_benford_law(df, column='QT_VOTOS'):
    """
    Aplica a Lei de Benford para identificar distribuições estatisticamente atípicas.
    A Lei de Benford descreve a distribuição esperada do primeiro dígito em conjuntos de dados naturais.
    Retorna um DataFrame com a distribuição observada vs esperada.
    """
    # Garantir que estamos trabalhando com números positivos
    df_positive = df[df[column] > 0].copy()
    
    # Extrair o primeiro dígito de cada valor
    df_positive['PRIMEIRO_DIGITO'] = df_positive[column].astype(str).str[0].astype(int)
    
    # Calcular a distribuição observada
    observed_counts = df_positive['PRIMEIRO_DIGITO'].value_counts().sort_index()
    observed_freq = observed_counts / observed_counts.sum()
    
    # Distribuição esperada pela Lei de Benford (logarítmica)
    expected_freq = pd.Series(
        {d: np.log10(1 + 1/d) for d in range(1, 10)},
        index=range(1, 10)
    )
    
    # Criar DataFrame para comparação
    benford_df = pd.DataFrame({
        'DIGITO': range(1, 10),
        'FREQ_OBSERVADA': observed_freq,
        'FREQ_ESPERADA': expected_freq
    })
    
    # Calcular a diferença entre observado e esperado
    benford_df['DIFERENCA'] = benford_df['FREQ_OBSERVADA'] - benford_df['FREQ_ESPERADA']
    benford_df['DIFERENCA_PCT'] = (benford_df['DIFERENCA'] / benford_df['FREQ_ESPERADA']) * 100
    
    # Calcular estatística chi-quadrado para teste de ajuste
    chi2 = sum((observed_freq - expected_freq)**2 / expected_freq)
    dof = len(expected_freq) - 1  # graus de liberdade
    p_value = 1 - stats.chi2.cdf(chi2, dof)
    
    # Adicionar estatísticas do teste
    benford_stats = {
        'chi2': chi2,
        'p_value': p_value,
        'benford_compliant': p_value > 0.05  # Compliance com Lei de Benford em nível 5%
    }
    
    return benford_df, benford_stats

def analyze_benford_law_advanced(df, column='QT_VOTOS', filter_min=10, agrupamento=None):
    """
    Realiza uma análise avançada da Lei de Benford nos dados, com testes estatísticos e visualizações.
    
    A Lei de Benford estabelece que em muitos conjuntos de dados numéricos, o primeiro dígito
    segue uma distribuição logarítmica, com o dígito 1 aparecendo cerca de 30% das vezes,
    enquanto o 9 aparece menos de 5% das vezes.
    
    Desvios significativos dessa distribuição podem indicar processos estatísticos não aleatórios
    ou características particulares do conjunto de dados analisado.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame contendo os dados
    column : str
        Nome da coluna para analisar
    filter_min : int
        Valor mínimo para filtrar os dados
    agrupamento : str, opcional
        Coluna para agrupar os dados antes da análise (ex: 'NM_MUNICIPIO')
        
    Returns:
    --------
    dict
        Dicionário com os resultados da análise
    """
    import numpy as np
    import pandas as pd
    import scipy.stats as stats
    import plotly.graph_objects as go
    import plotly.subplots as sp
    
    results = {}
    
    if column not in df.columns:
        return {"error": f"Coluna {column} não encontrada no DataFrame"}
    
    # Filtrar valores positivos acima do mínimo
    df_filtered = df[df[column] >= filter_min].copy()
    
    if len(df_filtered) == 0:
        return {"error": f"Não há dados suficientes após filtrar valores >= {filter_min}"}
    
    if agrupamento and agrupamento in df.columns:
        # Agrupar os dados pela coluna especificada
        grouped = df.groupby(agrupamento)[column].sum().reset_index()
        df_to_analyze = grouped
    else:
        df_to_analyze = df_filtered
    
    # Extrair o primeiro dígito
    df_to_analyze['first_digit'] = df_to_analyze[column].astype(str).str[0].astype(int)
    
    # Contar a frequência de cada primeiro dígito
    observed_counts = df_to_analyze['first_digit'].value_counts().sort_index()
    total_obs = observed_counts.sum()
    observed_freq = observed_counts / total_obs
    
    # Frequências esperadas pela Lei de Benford
    benford_freq = {
        1: 0.301, 2: 0.176, 3: 0.125, 4: 0.097, 
        5: 0.079, 6: 0.067, 7: 0.058, 8: 0.051, 9: 0.046
    }
    expected_freq = pd.Series(benford_freq)
    
    # Calcular a estatística do teste qui-quadrado
    expected_counts = total_obs * expected_freq
    chi2_stat = sum((observed_counts - expected_counts) ** 2 / expected_counts)
    dof = 8  # 9 categorias (dígitos 1-9) menos 1
    p_value = 1 - stats.chi2.cdf(chi2_stat, df=dof)
    
    # Resultados dos testes estatísticos
    results['total_observations'] = total_obs
    results['chi2_stat'] = chi2_stat
    results['p_value'] = p_value
    results['conformidade'] = p_value > 0.05  # Conformidade com a Lei de Benford (p > 0.05)
    
    # Calcular o Mean Absolute Deviation (MAD)
    mad = np.mean(np.abs(observed_freq - expected_freq))
    results['mad'] = mad
    
    # Interpretação do MAD
    if mad < 0.006:
        mad_interpretation = "Conformidade Próxima"
    elif mad < 0.012:
        mad_interpretation = "Conformidade Aceitável"
    elif mad < 0.015:
        mad_interpretation = "Conformidade Marginalmente Aceitável"
    else:
        mad_interpretation = "Não-Conformidade"
    
    results['mad_interpretation'] = mad_interpretation
    
    # Preparar dados para visualização
    digits = list(range(1, 10))
    observed = observed_freq.reindex(digits, fill_value=0).values
    expected = expected_freq.reindex(digits, fill_value=0).values
    
    results['observed_freq'] = observed
    results['expected_freq'] = expected
    results['digits'] = digits
    
    # Calcular o erro percentual para cada dígito
    error_pct = 100 * (observed - expected) / expected
    results['error_pct'] = error_pct
    
    return results

# Função para interface de análise de séries temporais
def temporal_analysis(df, date_col, value_col, group_col=None):
    """
    Realiza análise temporal dos dados.
    date_col: coluna com a data
    value_col: coluna com o valor a ser analisado
    group_col: coluna opcional para agrupar os dados
    """
    if df[date_col].dtype == 'object':
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    
    # Preparar dados para série temporal
    if group_col:
        # Agrupar por data e grupo
        ts_data = df.groupby([pd.Grouper(key=date_col, freq='D'), group_col])[value_col].sum().reset_index()
        
        # Gráfico por grupo ao longo do tempo
        fig = px.line(
            ts_data, 
            x=date_col, 
            y=value_col, 
            color=group_col,
            title=f'Evolução de {value_col} por {group_col} ao longo do tempo',
            labels={value_col: value_col, date_col: 'Data'}
        )
    else:
        # Agrupar apenas por data
        ts_data = df.groupby(pd.Grouper(key=date_col, freq='D'))[value_col].sum().reset_index()
        
        # Gráfico de série temporal
        fig = px.line(
            ts_data, 
            x=date_col, 
            y=value_col,
            title=f'Evolução de {value_col} ao longo do tempo',
            labels={value_col: value_col, date_col: 'Data'}
        )
    
    return fig, ts_data

def apply_autoencoder(df, columns=['QT_VOTOS', 'QT_ELEITORES'], contamination=0.01):
    """
    Aplica um autoencoder para detecção de anomalias.
    Retorna o DataFrame com uma coluna 'ANOMALIA_AUTOENCODER' booleana.
    """
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense
    except ImportError:
        return df.copy()
    
    df_copy = df.copy()
    
    # Preparar os dados para o autoencoder
    df_subset = df_copy.dropna(subset=columns)
    if len(df_subset) < 50:  # Se não tiver dados suficientes
        return df_copy
    
    # Normalizar os dados
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_subset[columns])
    
    # Definir a arquitetura do autoencoder
    input_dim = len(columns)
    encoding_dim = max(1, input_dim // 2)  # Camada de codificação
    
    # Construir o modelo
    autoencoder = Sequential([
        # Encoder
        Dense(encoding_dim, activation='relu', input_shape=(input_dim,)),
        # Decoder
        Dense(input_dim, activation='sigmoid')
    ])
    
    # Compilar o modelo
    autoencoder.compile(optimizer='adam', loss='mse')
    
    # Treinar o modelo
    autoencoder.fit(
        X_scaled, X_scaled,
        epochs=50,
        batch_size=32,
        shuffle=True,
        verbose=0
    )
    
    # Obter as previsões e calcular o erro de reconstrução
    X_pred = autoencoder.predict(X_scaled)
    mse = np.mean(np.power(X_scaled - X_pred, 2), axis=1)
    
    # Determinar um limiar para anomalias
    threshold = np.percentile(mse, 100 * (1 - contamination))
    
    # Marcar anomalias
    df_subset['ANOMALIA_AUTOENCODER'] = mse > threshold
    df_subset['MSE_AUTOENCODER'] = mse
    
    # Mesclar de volta com o DataFrame original
    df_copy.loc[df_subset.index, 'ANOMALIA_AUTOENCODER'] = df_subset['ANOMALIA_AUTOENCODER'].values
    df_copy.loc[df_subset.index, 'MSE_AUTOENCODER'] = df_subset['MSE_AUTOENCODER'].values
    
    # Preencher NaNs
    df_copy['ANOMALIA_AUTOENCODER'] = df_copy['ANOMALIA_AUTOENCODER'].fillna(False)
    
    return df_copy

def model_explanation(df, model_type, features, target_column=None):
    """
    Gera explicações para modelos de machine learning usando SHAP.
    """
    try:
        import shap
    except ImportError:
        return None, "Biblioteca SHAP não instalada."
    
    # Preparar dados
    X = df[features]
    
    # Aplicar modelo conforme o tipo
    if model_type == 'isolation_forest':
        model = IsolationForest(n_estimators=100, contamination='auto', random_state=42)
        model.fit(X)
        
        # Criar explicador SHAP
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        
        # Plotar resumo
        shap.summary_plot(shap_values, X, plot_type="bar", show=False)
        fig_importance = plt.gcf()
        plt.close()
        
        # Plotar dependência para a feature mais importante
        feature_idx = np.argmax(np.abs(shap_values).mean(axis=0))
        most_important_feature = features[feature_idx]
        shap.dependence_plot(most_important_feature, shap_values, X, show=False)
        fig_dependence = plt.gcf()
        plt.close()
        
        return {'importance': fig_importance, 'dependence': fig_dependence}, most_important_feature
    
    elif model_type == 'xgboost' and target_column is not None:
        import xgboost as xgb
        
        # Preparar target
        y = df[target_column]
        
        # Criar e treinar modelo XGBoost
        model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
        model.fit(X, y)
        
        # Criar explicador SHAP
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        
        # Plotar resumo
        shap.summary_plot(shap_values, X, plot_type="bar", show=False)
        fig_importance = plt.gcf()
        plt.close()
        
        # Plotar dependência para a feature mais importante
        feature_idx = np.argmax(np.abs(shap_values).mean(axis=0))
        most_important_feature = features[feature_idx]
        shap.dependence_plot(most_important_feature, shap_values, X, show=False)
        fig_dependence = plt.gcf()
        plt.close()
        
        return {'importance': fig_importance, 'dependence': fig_dependence}, most_important_feature
    
    else:
        return None, "Tipo de modelo não suportado."

def create_geo_map(df, anomaly_col=None, lat_col=None, lon_col=None):
    """
    Cria um mapa interativo com os dados geográficos, destacando anomalias.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame com os dados
    anomaly_col : str, optional
        Nome da coluna que indica se o registro é uma anomalia
    lat_col : str, optional
        Nome da coluna com a latitude
    lon_col : str, optional
        Nome da coluna com a longitude
    
    Returns:
    --------
    folium.Map
        Mapa com os pontos plotados
    """
    try:
        import folium
        from folium.plugins import MarkerCluster
        from folium.plugins import HeatMap
    except ImportError:
        return None, "As bibliotecas folium ou streamlit-folium não estão instaladas."
    
    # Verificar se temos coordenadas
    if lat_col is None or lon_col is None:
        # Se não temos coordenadas explícitas, tentamos usar dados geográficos do Brasil
        # Verificamos se temos a coluna de UF para agrupar os dados
        if 'SG_UF' in df.columns:
            # Coordenadas aproximadas dos centros dos estados brasileiros
            uf_coords = {
                'AC': [-9.0238, -70.812], 'AL': [-9.5713, -36.7819], 'AM': [-3.4168, -65.8561],
                'AP': [1.4493, -51.2146], 'BA': [-12.9718, -38.5011], 'CE': [-3.7172, -38.5433],
                'DF': [-15.7998, -47.8645], 'ES': [-19.1834, -40.3089], 'GO': [-16.6864, -49.2643],
                'MA': [-2.55, -44.3066], 'MG': [-18.5122, -44.555], 'MS': [-20.4428, -54.6464],
                'MT': [-12.6819, -56.9211], 'PA': [-5.5308, -52.2906], 'PB': [-7.115, -34.8631],
                'PE': [-8.4116, -37.0731], 'PI': [-8.0852, -42.7983], 'PR': [-25.2521, -52.0215],
                'RJ': [-22.9068, -43.1729], 'RN': [-5.7945, -36.2077], 'RO': [-11.5057, -63.5806],
                'RR': [2.7376, -62.0751], 'RS': [-30.0346, -51.2177], 'SC': [-27.2423, -50.2189],
                'SE': [-10.9472, -37.0731], 'SP': [-23.5505, -46.6333], 'TO': [-10.1753, -48.2982]
            }
            
            # Agrupar dados por UF e contar anomalias por UF
            if anomaly_col in df.columns:
                uf_data = df.groupby('SG_UF')[anomaly_col].agg(['count', 'sum']).reset_index()
                uf_data.columns = ['SG_UF', 'total', 'anomalias']
                uf_data['pct_anomalias'] = (uf_data['anomalias'] / uf_data['total'] * 100).round(2)
                
                # Adicionar coordenadas
                uf_data['lat'] = uf_data['SG_UF'].map(lambda x: uf_coords.get(x, [0, 0])[0])
                uf_data['lon'] = uf_data['SG_UF'].map(lambda x: uf_coords.get(x, [0, 0])[1])
                
                # Criar mapa
                map_center = [-15.7998, -47.8645]  # Centro do Brasil (aproximadamente DF)
                m = folium.Map(location=map_center, zoom_start=4, tiles='CartoDB positron')
                
                # Adicionar marcadores com informações
                for idx, row in uf_data.iterrows():
                    if row['anomalias'] > 0:
                        folium.CircleMarker(
                            location=[row['lat'], row['lon']],
                            radius=min(row['pct_anomalias'] * 2, 20),  # Tamanho baseado na % de anomalias
                            color='red',
                            fill=True,
                            fill_color='red',
                            fill_opacity=0.6,
                            popup=f"""
                            <div style='width: 150px'>
                                <b>{row['SG_UF']}</b><br>
                                Total: {row['total']}<br>
                                Anomalias: {row['anomalias']}<br>
                                % Anomalias: {row['pct_anomalias']}%
                            </div>
                            """
                        ).add_to(m)
                    else:
                        folium.CircleMarker(
                            location=[row['lat'], row['lon']],
                            radius=5,
                            color='blue',
                            fill=True,
                            fill_color='blue',
                            fill_opacity=0.6,
                            popup=f"""
                            <div style='width: 150px'>
                                <b>{row['SG_UF']}</b><br>
                                Total: {row['total']}<br>
                                Sem anomalias detectadas
                            </div>
                            """
                        ).add_to(m)
                
                return m, None
            else:
                # Se não temos coluna de anomalia, apenas criamos um mapa com os dados por UF
                uf_data = df.groupby('SG_UF').size().reset_index(name='total')
                
                # Adicionar coordenadas
                uf_data['lat'] = uf_data['SG_UF'].map(lambda x: uf_coords.get(x, [0, 0])[0])
                uf_data['lon'] = uf_data['SG_UF'].map(lambda x: uf_coords.get(x, [0, 0])[1])
                
                # Criar mapa
                map_center = [-15.7998, -47.8645]  # Centro do Brasil (aproximadamente DF)
                m = folium.Map(location=map_center, zoom_start=4, tiles='CartoDB positron')
                
                # Adicionar marcadores com informações
                for idx, row in uf_data.iterrows():
                    folium.CircleMarker(
                        location=[row['lat'], row['lon']],
                        radius=min(row['total'] / 1000, 15),  # Tamanho baseado no total
                        color='blue',
                        fill=True,
                        fill_color='blue',
                        fill_opacity=0.6,
                        popup=f"""
                        <div style='width: 150px'>
                            <b>{row['SG_UF']}</b><br>
                            Total: {row['total']}
                        </div>
                        """
                    ).add_to(m)
                
                return m, None
        else:
            return None, "Não foi possível criar o mapa: não foram encontradas coordenadas ou coluna de UF."
    else:
        # Se temos latitude e longitude, usamos essas coordenadas
        # Filtrar apenas registros com coordenadas válidas
        df_map = df.dropna(subset=[lat_col, lon_col]).copy()
        
        if len(df_map) == 0:
            return None, "Não há coordenadas válidas disponíveis."
        
        # Calcular centro do mapa
        map_center = [df_map[lat_col].mean(), df_map[lon_col].mean()]
        
        # Criar mapa
        m = folium.Map(location=map_center, zoom_start=5, tiles='CartoDB positron')
        
        # Adicionar cluster de marcadores
        marker_cluster = MarkerCluster().add_to(m)
        
        # Verificar se temos coluna de anomalia
        if anomaly_col in df_map.columns:
            # Adicionar marcadores para anomalias
            for idx, row in df_map[df_map[anomaly_col]].iterrows():
                folium.Marker(
                    location=[row[lat_col], row[lon_col]],
                    popup=f"""
                    <div style='width: 150px'>
                        <b>Anomalia detectada</b><br>
                        {row.get('NM_MUNICIPIO', 'Local')}<br>
                        Votos: {row.get('QT_VOTOS', 'N/A')}<br>
                    </div>
                    """,
                    icon=folium.Icon(color='red', icon='info-sign')
                ).add_to(marker_cluster)
            
            # Adicionar mapa de calor apenas com anomalias
            heat_data = df_map[df_map[anomaly_col]][[lat_col, lon_col]].values.tolist()
            if heat_data:
                HeatMap(heat_data, radius=15).add_to(m)
        
        return m, None

def calculate_anomaly_score(df, col_votos='QT_VOTOS', col_eleitores='QT_ELEITORES'):
    """
    Calcula um score de anomalia entre 0 e 100 com base em vários indicadores.
    Quanto maior o score, mais provável que seja uma anomalia significativa.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame contendo os dados
    col_votos : str
        Nome da coluna com a quantidade de votos
    col_eleitores : str
        Nome da coluna com a quantidade de eleitores
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame original com uma coluna adicional 'SCORE_ANOMALIA'
    """
    df_result = df.copy()
    
    # Inicializar pontuação
    df_result['SCORE_ANOMALIA'] = 0
    
    # 1. Componente: Z-Score (desvio em relação à média)
    if col_votos in df_result.columns:
        # Calcular Z-Score
        mean_val = df_result[col_votos].mean()
        std_val = df_result[col_votos].std()
        
        if std_val > 0:
            df_result['Z_SCORE_TEMP'] = abs((df_result[col_votos] - mean_val) / std_val)
            
            # Contribuição do Z-Score (até 40 pontos)
            # Normalizar para 0-40 pontos usando um limite superior de 5 desvios padrão
            df_result['SCORE_ANOMALIA'] += 40 * df_result['Z_SCORE_TEMP'].clip(0, 5) / 5
            
            # Limpar coluna temporária
            df_result.drop('Z_SCORE_TEMP', axis=1, inplace=True)
    
    # 2. Componente: Razão votos/eleitores (até 30 pontos)
    if col_votos in df_result.columns and col_eleitores in df_result.columns:
        # Evitar divisão por zero
        df_result['RAZAO_TEMP'] = df_result[col_votos] / df_result[col_eleitores].replace(0, np.nan)
        
        # Verificar se a razão é muito alta (> 0.9) ou muito baixa (< 0.1)
        # Valores extremos aumentam o score
        df_result['SCORE_ANOMALIA'] += np.where(
            df_result['RAZAO_TEMP'] > 0.95,  # Muito próximo de 100% de comparecimento
            30 * (df_result['RAZAO_TEMP'] - 0.95) * 20,  # Pontuação escalando até 30 pontos
            0
        )
        
        df_result['SCORE_ANOMALIA'] += np.where(
            (df_result['RAZAO_TEMP'] < 0.1) & (df_result[col_eleitores] > 100),  # Comparecimento muito baixo
            30 * (0.1 - df_result['RAZAO_TEMP']),  # Pontuação escalando até 30 pontos
            0
        )
        
        # Limpar coluna temporária
        df_result.drop('RAZAO_TEMP', axis=1, inplace=True)
    
    # 3. Componente: Tamanho da seção (até 15 pontos)
    if col_eleitores in df_result.columns:
        # Penalizar seções muito grandes ou muito pequenas
        mean_eleitores = df_result[col_eleitores].mean()
        std_eleitores = df_result[col_eleitores].std()
        
        if std_eleitores > 0:
            df_result['SIZE_ZSCORE'] = abs((df_result[col_eleitores] - mean_eleitores) / std_eleitores)
            
            # Contribuição do tamanho (até 15 pontos)
            # Normalizar para 0-15 pontos usando um limite superior de 3 desvios padrão
            df_result['SCORE_ANOMALIA'] += 15 * df_result['SIZE_ZSCORE'].clip(0, 3) / 3
            
            # Limpar coluna temporária
            df_result.drop('SIZE_ZSCORE', axis=1, inplace=True)
    
    # 4. Componente: Resultados de algoritmos de detecção (até 15 pontos)
    anomaly_cols = [col for col in df_result.columns if 'ANOMALIA_' in col]
    if anomaly_cols:
        # Contabilizar quantos algoritmos detectaram como anomalia
        df_result['ANOMALY_COUNT'] = 0
        for col in anomaly_cols:
            df_result['ANOMALY_COUNT'] += df_result[col].astype(int)
        
        # Contribuição dos algoritmos (até 15 pontos, 3 pontos por algoritmo que detectou)
        df_result['SCORE_ANOMALIA'] += 3 * df_result['ANOMALY_COUNT'].clip(0, 5)
        
        # Limpar coluna temporária
        df_result.drop('ANOMALY_COUNT', axis=1, inplace=True)
    
    # Garantir que o score esteja entre 0 e 100
    df_result['SCORE_ANOMALIA'] = df_result['SCORE_ANOMALIA'].clip(0, 100).round(1)
    
    # Adicionar categoria de severidade
    df_result['SEVERIDADE'] = pd.cut(
        df_result['SCORE_ANOMALIA'],
        bins=[0, 20, 40, 60, 80, 100],
        labels=['Muito Baixa', 'Baixa', 'Média', 'Alta', 'Muito Alta']
    )
    
    return df_result

def analyze_candidate_dominance(df, candidate_col='NM_VOTAVEL', vote_col='QT_VOTOS', section_id_cols=['NR_ZONA', 'NR_SECAO']):
    """
    Analisa a dominância de candidatos em seções eleitorais, identificando sessões onde
    um candidato recebeu uma concentração anormalmente alta de votos.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame com os dados eleitorais
    candidate_col : str
        Nome da coluna que contém o nome do candidato
    vote_col : str
        Nome da coluna que contém a quantidade de votos
    section_id_cols : list
        Lista de colunas que identificam uma seção eleitoral
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame com as seções, o total de votos, votos do candidato dominante,
        percentual de dominância e valor p para a hipótese de distribuição uniforme
    """
    # Verificar se temos as colunas necessárias
    required_cols = [candidate_col, vote_col] + section_id_cols
    if not all(col in df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df.columns]
        return None, f"Colunas necessárias ausentes: {', '.join(missing)}"
    
    # Agrupar por seção e candidato para obter votos por candidato por seção
    df_section_votes = df.groupby(section_id_cols + [candidate_col])[vote_col].sum().reset_index()
    
    # Calcular total de votos por seção
    section_totals = df_section_votes.groupby(section_id_cols)[vote_col].sum().reset_index()
    section_totals.rename(columns={vote_col: 'TOTAL_VOTOS_SECAO'}, inplace=True)
    
    # Encontrar o candidato dominante em cada seção
    dominant_candidates = df_section_votes.loc[df_section_votes.groupby(section_id_cols)[vote_col].idxmax()]
    dominant_candidates.rename(columns={vote_col: 'VOTOS_CANDIDATO_DOMINANTE'}, inplace=True)
    
    # Mesclar com totais da seção
    result = pd.merge(dominant_candidates, section_totals, on=section_id_cols)
    
    # Calcular percentual de dominância
    result['PERCENTUAL_DOMINANCIA'] = (result['VOTOS_CANDIDATO_DOMINANTE'] / result['TOTAL_VOTOS_SECAO'] * 100).round(2)
    
    # Calcular número de candidatos únicos por seção
    candidates_per_section = df_section_votes.groupby(section_id_cols)[candidate_col].nunique().reset_index()
    candidates_per_section.rename(columns={candidate_col: 'NUM_CANDIDATOS'}, inplace=True)
    result = pd.merge(result, candidates_per_section, on=section_id_cols)
    
    # Calcular valor p para teste de hipótese (distribuição uniforme vs. observada)
    # Hipótese nula: votos são distribuídos uniformemente entre candidatos
    result['VALOR_P'] = result.apply(
        lambda row: 1 - stats.binom.cdf(
            row['VOTOS_CANDIDATO_DOMINANTE'] - 1,  # -1 porque queremos P(X >= x)
            row['TOTAL_VOTOS_SECAO'],
            1 / row['NUM_CANDIDATOS']  # probabilidade sob H0 (uniforme)
        ),
        axis=1
    )
    
    # Ordenar pelo percentual de dominância (decrescente)
    result = result.sort_values('PERCENTUAL_DOMINANCIA', ascending=False)
    
    # Adicionar flags para seções com dominância extrema
    result['DOMINANCIA_SUSPEITA'] = result['PERCENTUAL_DOMINANCIA'] >= 95  # Mais de 95% dos votos
    result['DOMINANCIA_EXTREMA'] = result['PERCENTUAL_DOMINANCIA'] >= 99.9  # Praticamente 100%
    
    return result, None

def detect_statistical_patterns(df, vote_col='QT_VOTOS', candidate_col='NM_VOTAVEL', 
                           section_id_cols=['NR_ZONA', 'NR_SECAO'], min_votes=50):
    """
    Identifica padrões estatísticos atípicos nas seções eleitorais:
    1. Seções com concentração extremamente alta de votos para um único candidato
    2. Seções com taxas de participação significativamente acima ou abaixo da média
    3. Seções com taxa de abstenção estatisticamente incomum (muito baixa)
    4. Padrões de distribuição não usual (múltiplos de 10, 100, etc.)
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame com os dados eleitorais
    vote_col : str
        Nome da coluna que contém a quantidade de votos
    candidate_col : str
        Nome da coluna que contém o nome do candidato
    section_id_cols : list
        Lista de colunas que identificam uma seção eleitoral
    min_votes : int
        Número mínimo de votos em uma seção para ser considerada na análise
    
    Returns:
    --------
    dict
        Dicionário com os resultados das análises
    """
    results = {}
    
    # Verificar se temos as colunas necessárias
    required_cols = [vote_col, candidate_col] + section_id_cols
    if not all(col in df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df.columns]
        return {"error": f"Colunas necessárias ausentes: {', '.join(missing)}"}
    
    # 1. Identificar seções com dominância extrema (100% ou quase)
    try:
        # Agrupar votos por seção e candidato
        section_candidate_votes = df.groupby(section_id_cols + [candidate_col])[vote_col].sum().reset_index()
        
        # Total de votos por seção
        section_totals = section_candidate_votes.groupby(section_id_cols)[vote_col].sum().reset_index()
        section_totals = section_totals[section_totals[vote_col] >= min_votes]  # Filtrar seções com poucos votos
        section_totals.rename(columns={vote_col: 'TOTAL_VOTOS_SECAO'}, inplace=True)
        
        # Encontrar o candidato com mais votos em cada seção
        dominant_candidates = section_candidate_votes.loc[section_candidate_votes.groupby(section_id_cols)[vote_col].idxmax()]
        dominant_candidates.rename(columns={vote_col: 'VOTOS_CANDIDATO_DOMINANTE'}, inplace=True)
        
        # Mesclar com totais da seção
        dominance_df = pd.merge(dominant_candidates, section_totals, on=section_id_cols)
        
        # Calcular percentual de dominância
        dominance_df['PERCENTUAL_DOMINANCIA'] = (dominance_df['VOTOS_CANDIDATO_DOMINANTE'] / 
                                               dominance_df['TOTAL_VOTOS_SECAO'] * 100).round(2)
        
        # Identificar seções com dominância extrema (>=99%)
        extreme_dominance = dominance_df[dominance_df['PERCENTUAL_DOMINANCIA'] >= 99.0].copy()
        
        # Identificar especificamente seções com 100% para um candidato
        exact_100_pct = extreme_dominance[extreme_dominance['PERCENTUAL_DOMINANCIA'] == 100.0].copy()
        exact_100_pct = exact_100_pct.sort_values('TOTAL_VOTOS_SECAO', ascending=False)
        
        results['dominancia_extrema'] = extreme_dominance.sort_values('TOTAL_VOTOS_SECAO', ascending=False)
        results['dominancia_100_pct'] = exact_100_pct
    except Exception as e:
        results['error_dominancia'] = str(e)
    
    # 2. Análise de abstenção e participação
    try:
        if 'QT_ELEITORES' in df.columns:
            # Verificar se já temos os dados de dominância
            if 'dominancia_extrema' in results and 'error_dominancia' not in results:
                # Obter o número de eleitores por seção (único)
                eleitores_secao = df.groupby(section_id_cols)['QT_ELEITORES'].first().reset_index()
                
                # Mesclar com o total de votos por seção
                participacao = pd.merge(section_totals, eleitores_secao, on=section_id_cols)
                
                # Calcular taxa de participação e abstenção
                participacao['TAXA_PARTICIPACAO'] = (participacao['TOTAL_VOTOS_SECAO'] / 
                                                   participacao['QT_ELEITORES'] * 100).round(2)
                participacao['TAXA_ABSTENCAO'] = (100 - participacao['TAXA_PARTICIPACAO']).round(2)
                
                # Calcular estatísticas de participação
                participacao_media = participacao['TAXA_PARTICIPACAO'].mean()
                participacao_std = participacao['TAXA_PARTICIPACAO'].std()
                abstencao_media = participacao['TAXA_ABSTENCAO'].mean()
                abstencao_std = participacao['TAXA_ABSTENCAO'].std()
                
                # Identificar seções com participação anormal (±2.5 desvios padrão)
                participacao['PARTICIPACAO_ANORMAL'] = (
                    (participacao['TAXA_PARTICIPACAO'] > participacao_media + 2.5 * participacao_std) | 
                    (participacao['TAXA_PARTICIPACAO'] < participacao_media - 2.5 * participacao_std)
                )
                
                # Identificar seções com abstenção próxima de zero (menor que 5%)
                participacao['ABSTENCAO_QUASE_ZERO'] = participacao['TAXA_ABSTENCAO'] < 5.0
                zero_abstencao = participacao[participacao['ABSTENCAO_QUASE_ZERO']].copy()
                zero_abstencao = zero_abstencao.sort_values('TAXA_ABSTENCAO')
                
                # Identificar seções com abstenção muito alta (maior que 2 desvios padrão acima da média)
                participacao['ABSTENCAO_MUITO_ALTA'] = participacao['TAXA_ABSTENCAO'] > (abstencao_media + 2 * abstencao_std)
                alta_abstencao = participacao[participacao['ABSTENCAO_MUITO_ALTA']].copy()
                alta_abstencao = alta_abstencao.sort_values('TAXA_ABSTENCAO', ascending=False)
                
                # Filtrar apenas seções com participação anormal
                participacao_anormal = participacao[participacao['PARTICIPACAO_ANORMAL']].copy()
                participacao_anormal.sort_values('TAXA_PARTICIPACAO', ascending=False, inplace=True)
                
                results['participacao_anormal'] = participacao_anormal
                results['abstencao_quase_zero'] = zero_abstencao
                results['abstencao_muito_alta'] = alta_abstencao
                results['participacao_media'] = participacao_media
                results['participacao_std'] = participacao_std
                results['abstencao_media'] = abstencao_media
                results['abstencao_std'] = abstencao_std
    except Exception as e:
        results['error_participacao'] = str(e)
    
    # 3. Padrões "redondos" de votação (múltiplos de 10, 50, 100, etc.)
    try:
        # Verificamos se já temos os dados básicos carregados
        if 'error_dominancia' not in results and section_candidate_votes is not None:
            # Crie uma cópia para evitar modificar o DataFrame original
            round_patterns_df = section_candidate_votes.copy()
            
            # Adicione colunas para marcar números "redondos"
            round_patterns_df['IS_MULTIPLE_10'] = round_patterns_df[vote_col] % 10 == 0
            round_patterns_df['IS_MULTIPLE_50'] = round_patterns_df[vote_col] % 50 == 0
            round_patterns_df['IS_MULTIPLE_100'] = round_patterns_df[vote_col] % 100 == 0
            
            # Filtre apenas números redondos e com um valor mínimo significativo
            min_round_votes = 30  # Mínimo para considerar múltiplos interessantes
            round_patterns = round_patterns_df[
                (round_patterns_df['IS_MULTIPLE_10'] | 
                 round_patterns_df['IS_MULTIPLE_50'] | 
                 round_patterns_df['IS_MULTIPLE_100']) &
                (round_patterns_df[vote_col] >= min_round_votes)
            ].copy()
            
            # Adicione uma coluna para indicar o tipo de arredondamento
            round_patterns['TIPO_ARREDONDAMENTO'] = 'Múltiplo de 10'
            mask_50 = round_patterns['IS_MULTIPLE_50'] == True
            mask_100 = round_patterns['IS_MULTIPLE_100'] == True
            round_patterns.loc[mask_50, 'TIPO_ARREDONDAMENTO'] = 'Múltiplo de 50'
            round_patterns.loc[mask_100, 'TIPO_ARREDONDAMENTO'] = 'Múltiplo de 100'
            
            # Ordenar por valor de votos (decrescente)
            round_patterns = round_patterns.sort_values(vote_col, ascending=False)
            
            # Contar quantas seções têm padrões redondos
            round_patterns_count = len(round_patterns)
            round_patterns_pct = (round_patterns_count / len(section_candidate_votes)) * 100 if len(section_candidate_votes) > 0 else 0
            
            results['padroes_redondos'] = round_patterns
            results['padroes_redondos_count'] = round_patterns_count
            results['padroes_redondos_pct'] = round_patterns_pct
    except Exception as e:
        results['error_padroes_redondos'] = str(e)
    
    return results

def perform_section_clustering(df, features=['QT_VOTOS', 'QT_ELEITORES'], normalize=True, n_clusters=5):
    """
    Realiza clustering de seções eleitorais para identificar grupos com perfis semelhantes
    e detectar seções com comportamentos anômalos em relação ao seu grupo esperado.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame com os dados eleitorais
    features : list
        Lista de colunas a serem usadas como features para o clustering
    normalize : bool
        Se True, normaliza as features antes do clustering
    n_clusters : int
        Número de clusters a serem identificados
        
    Returns:
    --------
    tuple
        (df_with_clusters, model, metrics)
    """
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import silhouette_score, calinski_harabasz_score
    import numpy as np
    import pandas as pd
    
    # Verificar se temos as features necessárias
    missing_features = [f for f in features if f not in df.columns]
    if missing_features:
        return None, None, {"error": f"Features ausentes: {missing_features}"}
    
    # Preparar dados para clustering
    # Usar apenas dados numéricos e remover NaNs
    df_cluster = df.copy()
    X = df_cluster[features].fillna(0)
    
    # Normalizar dados se solicitado
    if normalize:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = X.values
    
    # Aplicar KMeans para clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df_cluster['CLUSTER'] = kmeans.fit_predict(X_scaled)
    
    # Calcular centróides dos clusters
    centroids = pd.DataFrame(
        scaler.inverse_transform(kmeans.cluster_centers_) if normalize else kmeans.cluster_centers_,
        columns=features
    )
    centroids['CLUSTER'] = range(n_clusters)
    
    # Calcular métricas de qualidade do clustering
    metrics = {}
    try:
        metrics['silhouette'] = silhouette_score(X_scaled, df_cluster['CLUSTER'])
        metrics['calinski_harabasz'] = calinski_harabasz_score(X_scaled, df_cluster['CLUSTER'])
    except:
        # Em caso de erro nas métricas (pode acontecer com poucos dados)
        metrics['silhouette'] = 0
        metrics['calinski_harabasz'] = 0
    
    # Calcular tamanho dos clusters
    cluster_sizes = df_cluster['CLUSTER'].value_counts().sort_index()
    metrics['cluster_sizes'] = cluster_sizes.to_dict()
    
    # Calcular distância de cada ponto ao centro do seu cluster
    # para identificar outliers dentro dos clusters
    distances = []
    
    for i, row in enumerate(X_scaled):
        cluster_id = df_cluster['CLUSTER'].iloc[i]
        centroid = kmeans.cluster_centers_[cluster_id]
        distance = np.linalg.norm(row - centroid)
        distances.append(distance)
    
    df_cluster['DISTANCIA_CENTROIDE'] = distances
    
    # Identificar outliers dentro de cada cluster (distância > 2 desvios padrão)
    df_cluster['OUTLIER_NO_CLUSTER'] = False
    for cluster_id in range(n_clusters):
        cluster_mask = df_cluster['CLUSTER'] == cluster_id
        cluster_distances = df_cluster.loc[cluster_mask, 'DISTANCIA_CENTROIDE']
        mean_dist = cluster_distances.mean()
        std_dist = cluster_distances.std()
        threshold = mean_dist + 2 * std_dist
        
        df_cluster.loc[cluster_mask & (df_cluster['DISTANCIA_CENTROIDE'] > threshold), 'OUTLIER_NO_CLUSTER'] = True
    
    # Estatísticas descritivas por cluster
    cluster_stats = df_cluster.groupby('CLUSTER')[features].agg(['mean', 'std', 'min', 'max'])
    metrics['cluster_stats'] = cluster_stats.to_dict()
    
    return df_cluster, kmeans, metrics, centroids

def detect_sequence_patterns(df, vote_col='QT_VOTOS', min_sequence=3, section_id_cols=['NR_ZONA', 'NR_SECAO']):
    """
    Identifica seções eleitorais com padrões de sequência numérica nas contagens de votos,
    o que pode indicar potencial fabricação de dados.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame com os dados eleitorais
    vote_col : str
        Nome da coluna que contém a quantidade de votos
    min_sequence : int
        Tamanho mínimo da sequência para ser considerada suspeita
    section_id_cols : list
        Lista de colunas que identificam uma seção eleitoral
        
    Returns:
    --------
    DataFrame
        DataFrame com as seções suspeitas
    """
    import pandas as pd
    import numpy as np
    
    results = {}
    
    # Verificar se temos as colunas necessárias
    required_cols = [vote_col] + section_id_cols
    if not all(col in df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df.columns]
        return {"error": f"Colunas necessárias ausentes: {', '.join(missing)}"}
    
    # Agrupar votos por seção
    section_votes = df.groupby(section_id_cols)[vote_col].apply(list).reset_index()
    section_votes.rename(columns={vote_col: 'LISTA_VOTOS'}, inplace=True)
    
    # Função para detectar sequências
    def has_sequence(votes_list, min_len=3):
        if len(votes_list) < min_len:
            return False, []
        
        # Ordenar votos para verificar sequências
        votes_sorted = sorted(votes_list)
        
        # Detectar sequências aritméticas (diferença constante)
        sequences = []
        
        for diff in range(1, 11):  # Verificar diferenças de 1 a 10
            current_seq = []
            
            for i in range(len(votes_sorted) - 1):
                if len(current_seq) == 0:
                    current_seq = [votes_sorted[i]]
                
                if votes_sorted[i+1] - votes_sorted[i] == diff:
                    current_seq.append(votes_sorted[i+1])
                else:
                    if len(current_seq) >= min_len:
                        sequences.append((current_seq.copy(), diff))
                    current_seq = [votes_sorted[i+1]]
            
            if len(current_seq) >= min_len:
                sequences.append((current_seq, diff))
        
        # Ordenar sequências por tamanho (decrescente)
        sequences.sort(key=lambda x: len(x[0]), reverse=True)
        
        return len(sequences) > 0, sequences
    
    # Aplicar detecção de sequências
    section_votes['TEM_SEQUENCIA'] = False
    section_votes['SEQUENCIAS'] = None
    section_votes['MAIOR_SEQUENCIA'] = 0
    section_votes['RAZAO_SEQUENCIA'] = 0
    
    for idx, row in section_votes.iterrows():
        has_seq, seqs = has_sequence(row['LISTA_VOTOS'], min_sequence)
        section_votes.at[idx, 'TEM_SEQUENCIA'] = has_seq
        
        if has_seq:
            section_votes.at[idx, 'SEQUENCIAS'] = str(seqs)
            section_votes.at[idx, 'MAIOR_SEQUENCIA'] = len(seqs[0][0])
            section_votes.at[idx, 'RAZAO_SEQUENCIA'] = seqs[0][1]
    
    # Filtrar apenas seções com sequências
    suspicious_sections = section_votes[section_votes['TEM_SEQUENCIA']].copy()
    
    # Ordenar por tamanho da maior sequência (decrescente)
    suspicious_sections.sort_values('MAIOR_SEQUENCIA', ascending=False, inplace=True)
    
    return suspicious_sections

def detect_blank_null_patterns(df, section_id_cols=['NR_ZONA', 'NR_SECAO'], min_voters=50):
    """
    Detecta padrões suspeitos de votos em branco e nulos nas seções eleitorais.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame com os dados eleitorais
    section_id_cols : list
        Lista de colunas que identificam uma seção eleitoral
    min_voters : int
        Número mínimo de eleitores para considerar uma seção na análise
        
    Returns:
    --------
    dict
        Dicionário com os resultados das análises
    """
    import pandas as pd
    import numpy as np
    
    results = {}
    
    # Verificar se temos as colunas necessárias
    required_cols = ['NM_VOTAVEL', 'QT_VOTOS', 'QT_ELEITORES'] + section_id_cols
    if not all(col in df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df.columns]
        return {"error": f"Colunas necessárias ausentes: {', '.join(missing)}"}
    
    try:
        # Identificar votos em branco e nulos
        branco_nulo_df = df[df['NM_VOTAVEL'].isin(['VOTO BRANCO', 'VOTO NULO'])].copy()
        
        if len(branco_nulo_df) == 0:
            return {"error": "Nenhum voto em branco ou nulo encontrado nos dados"}
        
        # Criar coluna para identificar o tipo de voto
        branco_nulo_df['TIPO_VOTO'] = branco_nulo_df['NM_VOTAVEL'].apply(lambda x: 'BRANCO' if x == 'VOTO BRANCO' else 'NULO')
        
        # Agrupar por seção e tipo de voto
        votos_por_secao_tipo = branco_nulo_df.groupby(section_id_cols + ['TIPO_VOTO'])['QT_VOTOS'].sum().reset_index()
        
        # Pivot para ter colunas de branco e nulo
        votos_pivot = votos_por_secao_tipo.pivot_table(
            index=section_id_cols, 
            columns='TIPO_VOTO', 
            values='QT_VOTOS',
            fill_value=0
        ).reset_index()
        
        # Verificar se temos as colunas esperadas
        if 'BRANCO' not in votos_pivot.columns:
            votos_pivot['BRANCO'] = 0
        if 'NULO' not in votos_pivot.columns:
            votos_pivot['NULO'] = 0
        
        # Adicionar total de eleitores
        total_eleitores = df.groupby(section_id_cols)['QT_ELEITORES'].first().reset_index()
        votos_pivot = pd.merge(votos_pivot, total_eleitores, on=section_id_cols)
        
        # Calcular total de votos válidos
        total_votos = df.groupby(section_id_cols)['QT_VOTOS'].sum().reset_index()
        votos_pivot = pd.merge(votos_pivot, total_votos, on=section_id_cols, suffixes=('', '_TOTAL'))
        
        # Calcular porcentagens
        votos_pivot['PCT_BRANCO'] = (votos_pivot['BRANCO'] / votos_pivot['QT_VOTOS_TOTAL'] * 100).round(2)
        votos_pivot['PCT_NULO'] = (votos_pivot['NULO'] / votos_pivot['QT_VOTOS_TOTAL'] * 100).round(2)
        votos_pivot['PCT_BRANCO_NULO'] = votos_pivot['PCT_BRANCO'] + votos_pivot['PCT_NULO']
        
        # Filtra apenas seções com número mínimo de eleitores
        votos_pivot = votos_pivot[votos_pivot['QT_ELEITORES'] >= min_voters]
        
        # Calcular estatísticas
        media_branco = votos_pivot['PCT_BRANCO'].mean()
        std_branco = votos_pivot['PCT_BRANCO'].std()
        media_nulo = votos_pivot['PCT_NULO'].mean()
        std_nulo = votos_pivot['PCT_NULO'].std()
        media_branco_nulo = votos_pivot['PCT_BRANCO_NULO'].mean()
        std_branco_nulo = votos_pivot['PCT_BRANCO_NULO'].std()
        
        # Identificar seções com valores anormais (> 2.5 desvios padrão)
        votos_pivot['BRANCO_ANORMAL'] = votos_pivot['PCT_BRANCO'] > (media_branco + 2.5 * std_branco)
        votos_pivot['NULO_ANORMAL'] = votos_pivot['PCT_NULO'] > (media_nulo + 2.5 * std_nulo)
        votos_pivot['BRANCO_NULO_ANORMAL'] = votos_pivot['PCT_BRANCO_NULO'] > (media_branco_nulo + 2.5 * std_branco_nulo)
        
        # Criar datasets filtrados
        alto_branco = votos_pivot[votos_pivot['BRANCO_ANORMAL']].sort_values('PCT_BRANCO', ascending=False)
        alto_nulo = votos_pivot[votos_pivot['NULO_ANORMAL']].sort_values('PCT_NULO', ascending=False)
        alto_branco_nulo = votos_pivot[votos_pivot['BRANCO_NULO_ANORMAL']].sort_values('PCT_BRANCO_NULO', ascending=False)
        
        # Armazenar resultados
        results['todas_secoes'] = votos_pivot
        results['alto_branco'] = alto_branco
        results['alto_nulo'] = alto_nulo
        results['alto_branco_nulo'] = alto_branco_nulo
        
        # Estatísticas
        results['media_branco'] = media_branco
        results['std_branco'] = std_branco
        results['media_nulo'] = media_nulo
        results['std_nulo'] = std_nulo
        results['media_branco_nulo'] = media_branco_nulo
        results['std_branco_nulo'] = std_branco_nulo
        
        results['contagem_alto_branco'] = len(alto_branco)
        results['contagem_alto_nulo'] = len(alto_nulo)
        results['contagem_alto_branco_nulo'] = len(alto_branco_nulo)
        
        return results
    
    except Exception as e:
        return {"error": str(e)}

def main():
    st.set_page_config(page_title="Dashboard - Análise de Eleições 2022", 
                      page_icon="📊", 
                      layout="wide")
    
    st.title("Dashboard - Análise Avançada dos Dados das Eleições 2022")
    st.markdown("""
    Este dashboard oferece análise avançada dos dados das eleições de 2022, com foco em:
    - **Detecção de Anomalias**: Identificação de padrões incomuns usando diversos métodos estatísticos e ML
    - **Análise Estatística**: Testes e métricas para avaliar distribuições e relações entre variáveis
    - **Visualizações**: Gráficos interativos para melhor compreensão dos dados
    """)
    
    # Detectar arquivos CSV disponíveis no diretório atual
    csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]
    parquet_files = [f for f in os.listdir('.') if f.endswith('.parquet')]
    
    if not csv_files:
        st.error("Nenhum arquivo CSV encontrado no diretório do projeto.")
        return
    
    # Usar o arquivo CSV encontrado ou o arquivo Parquet correspondente se existir
    csv_file = csv_files[0]  # Assume o primeiro arquivo CSV
    parquet_file = os.path.splitext(csv_file)[0] + '.parquet'
    
    # Mostrar qual arquivo será usado
    st.success(f"Arquivo detectado: {csv_file}")
    
    # Iniciar processamento imediato
    with st.spinner("Iniciando processamento do arquivo..."):
        # Verificar se o arquivo Parquet já existe
        if parquet_file in parquet_files:
            st.info(f"Usando arquivo Parquet existente: {parquet_file}")
            
            # Verificar se o arquivo CSV foi modificado depois do Parquet
            csv_time = os.path.getmtime(csv_file)
            parquet_time = os.path.getmtime(parquet_file)
            
            if csv_time > parquet_time:
                st.warning("O arquivo CSV foi modificado desde a última conversão. Reconvertendo...")
                try:
                    reduce_csv_size(csv_file, parquet_file)
                    st.success("Reconversão concluída!")
                except Exception as e:
                    st.error(f"Erro ao processar o arquivo: {str(e)}")
                    return
        else:
            st.info(f"Convertendo {csv_file} para formato Parquet. Isso pode levar alguns minutos...")
            try:
                reduce_csv_size(csv_file, parquet_file)
                st.success("Conversão concluída!")
            except Exception as e:
                st.error(f"Erro ao processar o arquivo: {str(e)}")
                return
    
    # Criar abas para organizar a interface
    tabs = st.tabs([
        "Dashboard de Padrões Atípicos", 
        "Visão Geral", 
        "Análise por Candidato",
        "Detecção de Anomalias", 
        "Análise Estatística", 
        "Análise Temporal",
        "Machine Learning",
        "Exportação"
    ])
    
    # Sidebar para filtros comuns a todas as abas
    st.sidebar.header("Filtros Globais")
    
    # Carregar dados uma única vez
    with st.spinner("Carregando dados..."):
        try:
            df = load_data(parquet_file)
            st.success(f"Dados carregados com sucesso! Total de registros: {len(df):,}")
        except Exception as e:
            st.error(f"Erro ao carregar os dados: {str(e)}")
            return
    
    # Filtros por UF na sidebar
    if 'SG_UF' in df.columns:
        ufs = sorted(df['SG_UF'].dropna().unique())
        uf = st.sidebar.selectbox("Selecione a UF", ufs)
        df_filtered = df[df['SG_UF'] == uf].copy()
        st.sidebar.write(f"Dados filtrados para a UF: {uf}")
    else:
        st.sidebar.error("Coluna 'SG_UF' não encontrada nos dados!")
        df_filtered = df.copy()
    
    # Filtro por Cargo
    if 'DS_CARGO' in df_filtered.columns:
        cargos = sorted(df_filtered['DS_CARGO'].dropna().unique())
        cargo = st.sidebar.selectbox("Selecione o Cargo", cargos)
        df_filtered = df_filtered[df_filtered['DS_CARGO'] == cargo].copy()
        st.sidebar.write(f"Dados filtrados para o cargo: {cargo}")
    
    # Filtro por quantidade de registros (amostragem)
    sample_size = st.sidebar.slider(
        "Limite de registros para análise", 
        min_value=1000, 
        max_value=min(100000, len(df_filtered)), 
        value=min(10000, len(df_filtered)),
        step=1000
    )
    
    if len(df_filtered) > sample_size:
        st.sidebar.info(f"Usando amostra de {sample_size} registros dos {len(df_filtered)} disponíveis")
        df_filtered = df_filtered.sample(sample_size, random_state=42)
    
    # Variáveis para análise
    col_votos = 'QT_VOTOS'  # Coluna principal de votos
    
    # NOVA ABA 1: DASHBOARD DE PADRÕES ATÍPICOS
    with tabs[0]:
        st.header("🔍 Dashboard de Padrões Estatísticos Atípicos")
        
        st.markdown("""
        Esta seção identifica automaticamente padrões estatísticos não usuais nas seções eleitorais:
        - **Concentração Extraordinária**: Seções onde um candidato recebeu percentual muito elevado dos votos (≥99%)
        - **Participação Não Usual**: Seções com taxas de comparecimento significativamente acima ou abaixo da média
        - **Abstenção Muito Baixa**: Seções com taxa de abstenção estatisticamente incomum (< 5%)
        - **Abstenção Muito Alta**: Seções com abstenção significativamente acima da média regional
        - **Distribuição Atípica**: Votações com números "redondos" (múltiplos de 10, 50, 100)
        """)
        
        # Botão para executar a análise
        if st.button("Identificar Padrões Estatísticos Atípicos"):
            with st.spinner("Analisando padrões estatísticos atípicos nas seções eleitorais..."):
                suspicious_patterns = detect_statistical_patterns(
                    df_filtered, 
                    vote_col='QT_VOTOS', 
                    candidate_col='NM_VOTAVEL', 
                    min_votes=50
                )
                
                if "error" in suspicious_patterns:
                    st.error(f"Erro na análise: {suspicious_patterns['error']}")
                else:
                    # Mostrar métricas resumidas
                    col_metrics1, col_metrics2, col_metrics3 = st.columns(3)
                    
                    with col_metrics1:
                        # Dominância extrema
                        if 'dominancia_extrema' in suspicious_patterns:
                            dominancia_df = suspicious_patterns['dominancia_extrema']
                            num_dominancia = len(dominancia_df)
                            
                            if num_dominancia > 0:
                                st.metric(
                                    "Seções com Dominância Extrema", 
                                    f"{num_dominancia}",
                                    help="Seções onde um candidato recebeu ≥99% dos votos"
                                )
                                
                                # Total de votantes afetados
                                total_votos_dominados = dominancia_df['TOTAL_VOTOS_SECAO'].sum()
                                st.caption(f"Total de votos: {total_votos_dominados:,}")
                                
                                # Exibir também seções com exatamente 100%
                                if 'dominancia_100_pct' in suspicious_patterns:
                                    num_100pct = len(suspicious_patterns['dominancia_100_pct'])
                                    if num_100pct > 0:
                                        st.caption(f"⚠️ {num_100pct} seções com *exatamente* 100% para um candidato")
                            else:
                                st.metric(
                                    "Seções com Dominância Extrema", 
                                    "0",
                                    help="Nenhuma seção com um único candidato recebendo ≥99% dos votos"
                                )
                    
                    with col_metrics2:
                        # Participação anormal
                        if 'participacao_anormal' in suspicious_patterns:
                            participacao_df = suspicious_patterns['participacao_anormal']
                            num_participacao = len(participacao_df)
                            
                            participacao_media = suspicious_patterns.get('participacao_media', 0)
                            participacao_std = suspicious_patterns.get('participacao_std', 0)
                            
                            if num_participacao > 0:
                                st.metric(
                                    "Seções com Participação Anormal", 
                                    f"{num_participacao}",
                                    help=f"Seções com participação muito acima ou abaixo da média ({participacao_media:.1f}±{2.5*participacao_std:.1f}%)"
                                )
                                
                                # Total de eleitores afetados
                                total_eleitores_anormais = participacao_df['QT_ELEITORES'].sum()
                                st.caption(f"Total de eleitores: {total_eleitores_anormais:,}")
                            else:
                                st.metric(
                                    "Seções com Participação Anormal", 
                                    "0",
                                    help="Nenhuma seção com taxa de participação anormal"
                                )
                        
                        # Exibir informações sobre abstenção quase zero
                        if 'abstencao_quase_zero' in suspicious_patterns:
                            num_zero_abstencao = len(suspicious_patterns['abstencao_quase_zero'])
                            if num_zero_abstencao > 0:
                                st.caption(f"⚠️ {num_zero_abstencao} seções com abstenção quase zero (<5%)")
                
                    with col_metrics3:
                        # Estatísticas gerais
                        if 'participacao_media' in suspicious_patterns:
                            participacao_media = suspicious_patterns['participacao_media']
                            st.metric(
                                "Taxa Média de Participação", 
                                f"{participacao_media:.1f}%",
                                help="Média de comparecimento em todas as seções"
                            )
                        
                        # Padrões redondos
                        if 'padroes_redondos_count' in suspicious_patterns:
                            num_redondos = suspicious_patterns['padroes_redondos_count']
                            pct_redondos = suspicious_patterns['padroes_redondos_pct']
                            
                            if num_redondos > 0:
                                st.metric(
                                    "Padrões Numéricos Redondos", 
                                    f"{num_redondos}",
                                    help=f"Candidatos com votações em múltiplos de 10, 50 ou 100"
                                )
                                st.caption(f"Representa {pct_redondos:.1f}% de todas as votações")
                
                    # Criar abas para mostrar detalhes
                    detail_tabs = st.tabs([
                        "Dominância Extrema", 
                        "Participação Anormal", 
                        "Abstenção Atípica",
                        "Padrões Redondos",
                        "Sequências Numéricas",
                        "Brancos e Nulos"  # Nova tab
                    ])
                    
                    # Aba de Dominância Extrema
                    with detail_tabs[0]:
                        if 'dominancia_extrema' in suspicious_patterns:
                            dominancia_df = suspicious_patterns['dominancia_extrema']
                            
                            if len(dominancia_df) > 0:
                                st.subheader("Seções com Dominância Extrema de Candidato")
                                
                                # Informação sobre dominância de 100%
                                if 'dominancia_100_pct' in suspicious_patterns:
                                    dominancia_100_df = suspicious_patterns['dominancia_100_pct']
                                    if len(dominancia_100_df) > 0:
                                        st.warning(f"⚠️ Foram encontradas {len(dominancia_100_df)} seções onde um candidato recebeu **100% dos votos**. Esta é uma situação rara e estatisticamente improvável, especialmente em seções com muitos eleitores.")
                                        
                                        # Opção para mostrar apenas seções com 100%
                                        mostrar_apenas_100 = st.checkbox("Mostrar apenas seções com 100% para um candidato")
                                        if mostrar_apenas_100:
                                            dominancia_df = dominancia_100_df
                            
                                # Adicionar município aos dados, se disponível
                                if 'NM_MUNICIPIO' in df_filtered.columns:
                                    municipios = df_filtered[['NR_ZONA', 'NR_SECAO', 'NM_MUNICIPIO']].drop_duplicates()
                                    dominancia_df = pd.merge(dominancia_df, municipios, on=['NR_ZONA', 'NR_SECAO'], how='left')
                                
                                # Mostrar tabela interativa
                                display_cols = [
                                    'NM_MUNICIPIO', 'NR_ZONA', 'NR_SECAO', 'NM_VOTAVEL', 
                                    'VOTOS_CANDIDATO_DOMINANTE', 'TOTAL_VOTOS_SECAO', 'PERCENTUAL_DOMINANCIA'
                                ]
                                display_cols = [col for col in display_cols if col in dominancia_df.columns]
                                
                                st.dataframe(
                                    dominancia_df[display_cols],
                                    column_config={
                                        'PERCENTUAL_DOMINANCIA': st.column_config.ProgressColumn(
                                            '% dos Votos',
                                            help='Percentual dos votos da seção recebidos pelo candidato',
                                            format="%.2f%%",
                                            min_value=0,
                                            max_value=100,
                                        )
                                    }
                                )
                                
                                # Gráfico de barras por candidato
                                candidato_counts = dominancia_df['NM_VOTAVEL'].value_counts().reset_index()
                                candidato_counts.columns = ['Candidato', 'Número de Seções']
                                
                                fig = px.bar(
                                    candidato_counts.head(10), 
                                    x='Candidato', 
                                    y='Número de Seções',
                                    title='Top 10 Candidatos com Dominância Extrema',
                                    labels={
                                        'Candidato': 'Candidato', 
                                        'Número de Seções': 'Número de Seções com Dominância Extrema'
                                    }
                                )
                                st.plotly_chart(fig)
                                
                                # Download dos dados
                                csv = dominancia_df.to_csv(index=False)
                                st.download_button(
                                    label="Baixar dados de dominância extrema como CSV",
                                    data=csv,
                                    file_name="secoes_dominancia_extrema.csv",
                                    mime="text/csv"
                                )
                            else:
                                st.info("Nenhuma seção com dominância extrema identificada.")
                    
                    # Aba de Participação Anormal
                    with detail_tabs[1]:
                        if 'participacao_anormal' in suspicious_patterns:
                            participacao_df = suspicious_patterns['participacao_anormal']
                            
                            if len(participacao_df) > 0:
                                st.subheader("Seções com Taxa de Participação Anormal")
                                
                                # Adicionar município aos dados, se disponível
                                if 'NM_MUNICIPIO' in df_filtered.columns:
                                    municipios = df_filtered[['NR_ZONA', 'NR_SECAO', 'NM_MUNICIPIO']].drop_duplicates()
                                    participacao_df = pd.merge(participacao_df, municipios, on=['NR_ZONA', 'NR_SECAO'], how='left')
                                
                                # Classificar participação como alta ou baixa
                                participacao_media = suspicious_patterns.get('participacao_media', 0)
                                participacao_df['TIPO_ANOMALIA'] = np.where(
                                    participacao_df['TAXA_PARTICIPACAO'] > participacao_media,
                                    'Alta (Suspeita)', 
                                    'Baixa (Abstenção)'
                                )
                                
                                # Mostrar tabela interativa
                                display_cols = [
                                    'NM_MUNICIPIO', 'NR_ZONA', 'NR_SECAO', 'TAXA_PARTICIPACAO', 
                                    'TIPO_ANOMALIA', 'TOTAL_VOTOS_SECAO', 'QT_ELEITORES'
                                ]
                                display_cols = [col for col in display_cols if col in participacao_df.columns]
                                
                                st.dataframe(
                                    participacao_df[display_cols],
                                    column_config={
                                        'TAXA_PARTICIPACAO': st.column_config.ProgressColumn(
                                            'Taxa de Participação',
                                            help='Percentual de comparecimento',
                                            format="%.2f%%",
                                            min_value=0,
                                            max_value=100,
                                        ),
                                        'TIPO_ANOMALIA': st.column_config.Column(
                                            'Tipo de Anomalia',
                                            help='Classificação da anomalia de participação',
                                            width='medium',
                                        )
                                    }
                                )
                                
                                # Histograma de participação
                                fig = px.histogram(
                                    participacao_df, 
                                    x='TAXA_PARTICIPACAO',
                                    color='TIPO_ANOMALIA',
                                    nbins=30,
                                    title='Distribuição das Taxas de Participação Anormais',
                                    labels={
                                        'TAXA_PARTICIPACAO': 'Taxa de Participação (%)',
                                        'count': 'Número de Seções',
                                        'TIPO_ANOMALIA': 'Tipo de Anomalia'
                                    }
                                )
                                
                                # Adicionar linha vertical para a média
                                fig.add_vline(
                                    x=participacao_media, 
                                    line_dash="dash", 
                                    line_color="black",
                                    annotation_text=f"Média: {participacao_media:.1f}%",
                                    annotation_position="top right"
                                )
                                
                                st.plotly_chart(fig)
                                
                                # Download dos dados
                                csv = participacao_df.to_csv(index=False)
                                st.download_button(
                                    label="Baixar dados de participação anormal como CSV",
                                    data=csv,
                                    file_name="secoes_participacao_anormal.csv",
                                    mime="text/csv"
                                )
                            else:
                                st.info("Nenhuma seção com participação anormal identificada.")
                    
                    # NOVA: Aba de Abstenção Atípica
                    with detail_tabs[2]:
                        st.subheader("Seções com Taxas de Abstenção Atípicas")
                        
                        abstencao_tabs = st.tabs(["Abstenção Quase Zero", "Abstenção Muito Alta"])
                        
                        # Abstenção Quase Zero
                        with abstencao_tabs[0]:
                            if 'abstencao_quase_zero' in suspicious_patterns:
                                zero_df = suspicious_patterns['abstencao_quase_zero']
                                
                                if len(zero_df) > 0:
                                    st.warning("""
                                    ⚠️ **Seções com abstenção quase zero são estatisticamente improváveis**
                                    
                                    Em eleições reais, sempre existe um percentual natural de abstenção devido a diversos fatores:
                                    - Eleitores que não podem comparecer (doença, viagem, etc.)
                                    - Cadastros desatualizados de eleitores falecidos ou que mudaram
                                    - Eleitores que simplesmente decidem não votar
                                    
                                    Taxas de abstenção abaixo de 5% são extremamente raras e podem indicar irregularidades.
                                    """)
                                    
                                    # Adicionar município aos dados, se disponível
                                    if 'NM_MUNICIPIO' in df_filtered.columns:
                                        municipios = df_filtered[['NR_ZONA', 'NR_SECAO', 'NM_MUNICIPIO']].drop_duplicates()
                                        zero_df = pd.merge(zero_df, municipios, on=['NR_ZONA', 'NR_SECAO'], how='left')
                                    
                                    # Mostrar tabela interativa
                                    display_cols = [
                                        'NM_MUNICIPIO', 'NR_ZONA', 'NR_SECAO', 'TAXA_ABSTENCAO',
                                        'TAXA_PARTICIPACAO', 'TOTAL_VOTOS_SECAO', 'QT_ELEITORES'
                                    ]
                                    display_cols = [col for col in display_cols if col in zero_df.columns]
                                    
                                    st.dataframe(
                                        zero_df[display_cols],
                                        column_config={
                                            'TAXA_ABSTENCAO': st.column_config.ProgressColumn(
                                                'Taxa de Abstenção',
                                                help='Percentual de não comparecimento',
                                                format="%.2f%%",
                                                min_value=0,
                                                max_value=100,
                                            ),
                                            'TAXA_PARTICIPACAO': st.column_config.ProgressColumn(
                                                'Taxa de Participação',
                                                help='Percentual de comparecimento',
                                                format="%.2f%%",
                                                min_value=0,
                                                max_value=100,
                                            )
                                        }
                                    )
                                    
                                    # Download dos dados
                                    csv = zero_df.to_csv(index=False)
                                    st.download_button(
                                        label="Baixar dados de abstenção quase zero como CSV",
                                        data=csv,
                                        file_name="secoes_abstencao_quase_zero.csv",
                                        mime="text/csv"
                                    )
                                else:
                                    st.info("Nenhuma seção com abstenção quase zero identificada.")
                        
                        # Abstenção Muito Alta
                        with abstencao_tabs[1]:
                            if 'abstencao_muito_alta' in suspicious_patterns:
                                alta_df = suspicious_patterns['abstencao_muito_alta']
                                abstencao_media = suspicious_patterns.get('abstencao_media', 0)
                                abstencao_std = suspicious_patterns.get('abstencao_std', 0)
                                
                                if len(alta_df) > 0:
                                    st.info(f"""
                                    ℹ️ Seções com abstenção muito acima da média regional
                                    
                                    A média de abstenção é {abstencao_media:.1f}% (±{abstencao_std:.1f}%)
                                    Estão listadas seções com abstenção acima de {(abstencao_media + 2*abstencao_std):.1f}%
                                    """)
                                    
                                    # Adicionar município aos dados, se disponível
                                    if 'NM_MUNICIPIO' in df_filtered.columns:
                                        municipios = df_filtered[['NR_ZONA', 'NR_SECAO', 'NM_MUNICIPIO']].drop_duplicates()
                                        alta_df = pd.merge(alta_df, municipios, on=['NR_ZONA', 'NR_SECAO'], how='left')
                                    
                                    # Mostrar tabela interativa
                                    display_cols = [
                                        'NM_MUNICIPIO', 'NR_ZONA', 'NR_SECAO', 'TAXA_ABSTENCAO',
                                        'TAXA_PARTICIPACAO', 'TOTAL_VOTOS_SECAO', 'QT_ELEITORES'
                                    ]
                                    display_cols = [col for col in display_cols if col in alta_df.columns]
                                    
                                    st.dataframe(
                                        alta_df[display_cols],
                                        column_config={
                                            'TAXA_ABSTENCAO': st.column_config.ProgressColumn(
                                                'Taxa de Abstenção',
                                                help='Percentual de não comparecimento',
                                                format="%.2f%%",
                                                min_value=0,
                                                max_value=100,
                                            ),
                                            'TAXA_PARTICIPACAO': st.column_config.ProgressColumn(
                                                'Taxa de Participação',
                                                help='Percentual de comparecimento',
                                                format="%.2f%%",
                                                min_value=0,
                                                max_value=100,
                                            )
                                        }
                                    )
                                    
                                    # Download dos dados
                                    csv = alta_df.to_csv(index=False)
                                    st.download_button(
                                        label="Baixar dados de abstenção muito alta como CSV",
                                        data=csv,
                                        file_name="secoes_abstencao_muito_alta.csv",
                                        mime="text/csv"
                                    )
                                else:
                                    st.info("Nenhuma seção com abstenção muito alta identificada.")
                    
                    # NOVA: Aba de Padrões Redondos
                    with detail_tabs[3]:
                        if 'padroes_redondos' in suspicious_patterns:
                            redondos_df = suspicious_patterns['padroes_redondos']
                            
                            if len(redondos_df) > 0:
                                st.subheader("Padrões 'Redondos' de Votação")
                                
                                st.markdown("""
                                ℹ️ **Sobre Padrões Redondos:**
                                
                                Em situações naturais, a distribuição do último dígito de contagens deve ser aproximadamente uniforme. 
                                Uma alta incidência de números "redondos" (múltiplos de 10, 50, 100) pode indicar números fabricados manualmente, 
                                já que humanos tendem a preferir números redondos ao estimar quantidades.
                                
                                Esta análise identifica votações que são múltiplos exatos de 10, 50 e 100, 
                                que ocorrem naturalmente com uma frequência muito menor do que outros valores.
                                """)
                                
                                # Adicionar município aos dados, se disponível
                                if 'NM_MUNICIPIO' in df_filtered.columns:
                                    municipios = df_filtered[['NR_ZONA', 'NR_SECAO', 'NM_MUNICIPIO']].drop_duplicates()
                                    redondos_df = pd.merge(redondos_df, municipios, on=['NR_ZONA', 'NR_SECAO'], how='left')
                                
                                # Mostrar tabela interativa
                                display_cols = [
                                    'NM_MUNICIPIO', 'NR_ZONA', 'NR_SECAO', 'NM_VOTAVEL', 
                                    'QT_VOTOS', 'TIPO_ARREDONDAMENTO'
                                ]
                                display_cols = [col for col in display_cols if col in redondos_df.columns]
                                
                                st.dataframe(
                                    redondos_df[display_cols].sort_values('QT_VOTOS', ascending=False),
                                    column_config={
                                        'TIPO_ARREDONDAMENTO': st.column_config.Column(
                                            'Tipo de Número Redondo',
                                            help='Categoria do número redondo',
                                            width='medium',
                                        )
                                    }
                                )
                                
                                # Contagem por tipo de arredondamento
                                arredondamento_counts = redondos_df['TIPO_ARREDONDAMENTO'].value_counts().reset_index()
                                arredondamento_counts.columns = ['Tipo', 'Contagem']
                                
                                fig = px.pie(
                                    arredondamento_counts, 
                                    values='Contagem', 
                                    names='Tipo',
                                    title='Distribuição de Padrões Redondos',
                                    color_discrete_sequence=px.colors.sequential.RdBu
                                )
                                st.plotly_chart(fig)
                                
                                # Download dos dados
                                csv = redondos_df.to_csv(index=False)
                                st.download_button(
                                    label="Baixar dados de padrões redondos como CSV",
                                    data=csv,
                                    file_name="secoes_padroes_redondos.csv",
                                    mime="text/csv"
                                )
                            else:
                                st.info("Nenhum padrão de números redondos identificado.")
                    
                    # Quando estamos na última tab (Sequências Numéricas):
                    with detail_tabs[4]:
                        st.subheader("🔢 Padrões de Sequência Numérica")
                        
                        st.markdown("""
                        Esta análise detecta seções eleitorais com votações que seguem sequências numéricas (progressões aritméticas),
                        como 10, 20, 30... ou 25, 50, 75...
                        
                        Tais padrões são estatisticamente improváveis em contagens naturais e podem indicar fabricação manual de dados.
                        """)
                        
                        # Botão para executar análise de sequências
                        if st.button("Detectar Sequências Numéricas"):
                            with st.spinner("Analisando padrões de sequência nas seções eleitorais..."):
                                min_sequence_length = 3  # Mínimo de 3 números em sequência para ser suspeito
                                
                                # Definir colunas que identificam uma seção
                                section_id_cols = ['NR_ZONA', 'NR_SECAO']
                                
                                sequences_df = detect_sequence_patterns(
                                    df_filtered, 
                                    vote_col='QT_VOTOS',
                                    min_sequence=min_sequence_length,
                                    section_id_cols=section_id_cols
                                )
                                
                                if isinstance(sequences_df, dict) and 'error' in sequences_df:
                                    st.error(f"Erro na análise: {sequences_df['error']}")
                                else:
                                    # Mostrar estatísticas
                                    n_suspicious = len(sequences_df)
                                    pct_suspicious = (n_suspicious / len(df_filtered[section_id_cols].drop_duplicates())) * 100
                                    
                                    st.metric(
                                        "Seções com Sequências Numéricas",
                                        f"{n_suspicious} ({pct_suspicious:.2f}%)"
                                    )
                                    
                                    if n_suspicious > 0:
                                        # Adicionar informações de município, se disponível
                                        if 'NM_MUNICIPIO' in df_filtered.columns:
                                            municipios = df_filtered[['NR_ZONA', 'NR_SECAO', 'NM_MUNICIPIO']].drop_duplicates()
                                            sequences_df = pd.merge(sequences_df, municipios, on=['NR_ZONA', 'NR_SECAO'], how='left')
                                        
                                        # Mostrar tabela com as seções suspeitas
                                        display_cols = ['NM_MUNICIPIO', 'NR_ZONA', 'NR_SECAO', 'MAIOR_SEQUENCIA', 'RAZAO_SEQUENCIA', 'LISTA_VOTOS']
                                        display_cols = [col for col in display_cols if col in sequences_df.columns]
                                        
                                        st.dataframe(
                                            sequences_df[display_cols],
                                            use_container_width=True,
                                            column_config={
                                                'MAIOR_SEQUENCIA': st.column_config.NumberColumn(
                                                    'Tamanho da Sequência',
                                                    help='Quantidade de números na maior sequência detectada',
                                                ),
                                                'RAZAO_SEQUENCIA': st.column_config.NumberColumn(
                                                    'Razão da Sequência',
                                                    help='Diferença constante entre os números da sequência',
                                                ),
                                                'LISTA_VOTOS': st.column_config.TextColumn(
                                                    'Votos na Seção',
                                                    help='Lista de contagens de votos presentes na seção',
                                                )
                                            }
                                        )
                                        
                                        # Download dos dados
                                        csv = sequences_df.to_csv(index=False)
                                        st.download_button(
                                            label="Baixar dados de sequências numéricas como CSV",
                                            data=csv,
                                            file_name="secoes_sequencias_numericas.csv",
                                            mime="text/csv"
                                        )
                                    else:
                                        st.info("Nenhuma seção com padrões de sequência numérica detectada.")
                                        
                                    st.markdown("""
                                    > **Nota:** A presença de sequências numéricas não é, por si só, evidência de fraude,
                                    > mas representa um padrão estatisticamente improvável que merece investigação adicional.
                                    > Em dados naturais, esperamos uma distribuição mais aleatória de contagens.
                                    """)
                    
                    # Com a última tab (Brancos e Nulos):
                    with detail_tabs[5]:
                        st.subheader("🗳️ Análise de Votos em Branco e Nulos")
                        
                        st.markdown("""
                        Esta análise identifica seções eleitorais com padrões suspeitos de votos em branco e nulos.
                        
                        Uma porcentagem anormalmente alta destes votos pode indicar:
                        - Problemas com as urnas ou instruções aos eleitores
                        - Protestos organizados dos eleitores
                        - Possível manipulação dos resultados
                        
                        A análise identifica seções com valores estatisticamente discrepantes (> 2.5 desvios padrão acima da média).
                        """)
                        
                        # Botão para executar análise
                        if st.button("Analisar Votos em Branco e Nulos"):
                            with st.spinner("Analisando padrões de votos em branco e nulos..."):
                                # Definir colunas que identificam uma seção
                                section_id_cols = ['NR_ZONA', 'NR_SECAO']
                                
                                # Opções de filtro
                                col_filt1, col_filt2 = st.columns(2)
                                with col_filt1:
                                    min_voters = st.slider(
                                        "Mínimo de eleitores por seção", 
                                        min_value=10, 
                                        max_value=500, 
                                        value=50, 
                                        help="Filtra seções com menos eleitores que o valor especificado"
                                    )
                                
                                # Executar a análise
                                blank_null_results = detect_blank_null_patterns(
                                    df_filtered,
                                    section_id_cols=section_id_cols,
                                    min_voters=min_voters
                                )
                                
                                if "error" in blank_null_results:
                                    st.error(f"Erro na análise: {blank_null_results['error']}")
                                else:
                                    # Resumo executivo
                                    st.info("""
                                    ### 📋 Resumo Executivo
                                    """)
                                    
                                    # Calcular estatísticas para o resumo
                                    total_secoes = len(blank_null_results['todas_secoes'])
                                    pct_alto_branco = blank_null_results['contagem_alto_branco'] / total_secoes * 100 if total_secoes > 0 else 0
                                    pct_alto_nulo = blank_null_results['contagem_alto_nulo'] / total_secoes * 100 if total_secoes > 0 else 0
                                    pct_alto_combinado = blank_null_results['contagem_alto_branco_nulo'] / total_secoes * 100 if total_secoes > 0 else 0
                                    
                                    # Texto do resumo
                                    resumo_text = f"""
                                    - **Total de seções analisadas**: {total_secoes}
                                    - **Votos em branco**: Média de {blank_null_results['media_branco']:.2f}% por seção, com desvio padrão de {blank_null_results['std_branco']:.2f}%
                                    - **Votos nulos**: Média de {blank_null_results['media_nulo']:.2f}% por seção, com desvio padrão de {blank_null_results['std_nulo']:.2f}%
                                    - **Combinado (brancos + nulos)**: Média de {blank_null_results['media_branco_nulo']:.2f}% por seção
                                    
                                    **Principais achados**:
                                    - {blank_null_results['contagem_alto_branco']} seções ({pct_alto_branco:.1f}% do total) apresentam percentual de votos em branco anormalmente alto
                                    - {blank_null_results['contagem_alto_nulo']} seções ({pct_alto_nulo:.1f}% do total) apresentam percentual de votos nulos anormalmente alto
                                    - {blank_null_results['contagem_alto_branco_nulo']} seções ({pct_alto_combinado:.1f}% do total) apresentam percentual combinado anormalmente alto
                                    
                                    **Recomendação**: As seções com valores anômalos devem ser investigadas prioritariamente.
                                    """
                                    
                                    st.markdown(resumo_text)
                                    
                                    # Mostrar métricas principais
                                    col1, col2, col3 = st.columns(3)
                                    
                                    with col1:
                                        st.metric(
                                            "Seções com Alto % de Brancos",
                                            f"{blank_null_results['contagem_alto_branco']} seções",
                                            help="Seções com percentual de votos em branco acima de 2.5 desvios padrão da média. Valores altos podem indicar problemas na urna ou protesto dos eleitores."
                                        )
                                        st.caption(f"Média: {blank_null_results['media_branco']:.2f}% | Desvio: {blank_null_results['std_branco']:.2f}%")
                                    
                                    with col2:
                                        st.metric(
                                            "Seções com Alto % de Nulos",
                                            f"{blank_null_results['contagem_alto_nulo']} seções",
                                            help="Seções com percentual de votos nulos acima de 2.5 desvios padrão da média. Valores altos podem indicar problemas técnicos ou dificuldade dos eleitores."
                                        )
                                        st.caption(f"Média: {blank_null_results['media_nulo']:.2f}% | Desvio: {blank_null_results['std_nulo']:.2f}%")
                                    
                                    with col3:
                                        st.metric(
                                            "Seções com Alto % Total",
                                            f"{blank_null_results['contagem_alto_branco_nulo']} seções",
                                            help="Seções com percentual combinado de brancos e nulos acima de 2.5 desvios padrão da média. Valores muito acima da média podem indicar anomalias significativas."
                                        )
                                        st.caption(f"Média: {blank_null_results['media_branco_nulo']:.2f}% | Desvio: {blank_null_results['std_branco_nulo']:.2f}%")
                                    
                                    # Criar sub-tabs para diferentes visualizações
                                    blank_null_subtabs = st.tabs([
                                        "Alto % de Brancos", 
                                        "Alto % de Nulos", 
                                        "Alto % Combinado",
                                        "Visualizações",
                                        "Comparativo"
                                    ])
                                    
                                    # Sub-tab 1: Alto % de Brancos
                                    with blank_null_subtabs[0]:
                                        if len(blank_null_results['alto_branco']) > 0:
                                            # Adicionar município aos dados, se disponível
                                            alto_branco_df = blank_null_results['alto_branco'].copy()
                                            if 'NM_MUNICIPIO' in df_filtered.columns:
                                                municipios = df_filtered[['NR_ZONA', 'NR_SECAO', 'NM_MUNICIPIO']].drop_duplicates()
                                                alto_branco_df = pd.merge(alto_branco_df, municipios, on=['NR_ZONA', 'NR_SECAO'], how='left')
                                            
                                            # Selecionar colunas para exibição
                                            display_cols = ['NM_MUNICIPIO', 'NR_ZONA', 'NR_SECAO', 'BRANCO', 'PCT_BRANCO', 'QT_VOTOS_TOTAL', 'QT_ELEITORES']
                                            display_cols = [col for col in display_cols if col in alto_branco_df.columns]
                                            
                                            st.dataframe(
                                                alto_branco_df[display_cols],
                                                use_container_width=True,
                                                column_config={
                                                    'PCT_BRANCO': st.column_config.NumberColumn(
                                                        '% Brancos',
                                                        help='Porcentagem de votos em branco',
                                                        format="%.2f%%"
                                                    ),
                                                    'BRANCO': st.column_config.NumberColumn(
                                                        'Votos Brancos',
                                                        help='Quantidade de votos em branco',
                                                        format="%d"
                                                    ),
                                                    'QT_VOTOS_TOTAL': st.column_config.NumberColumn(
                                                        'Total de Votos',
                                                        help='Total de votos na seção',
                                                        format="%d"
                                                    ),
                                                    'QT_ELEITORES': st.column_config.NumberColumn(
                                                        'Eleitores',
                                                        help='Total de eleitores na seção',
                                                        format="%d"
                                                    )
                                                }
                                            )
                                            
                                            # Download dos dados
                                            csv = alto_branco_df.to_csv(index=False)
                                            st.download_button(
                                                label="Baixar dados de seções com alto % de brancos",
                                                data=csv,
                                                file_name="secoes_alto_percentual_brancos.csv",
                                                mime="text/csv"
                                            )
                                        else:
                                            st.info("Nenhuma seção com percentual anormalmente alto de votos em branco.")
                                    
                                    # Sub-tab 2: Alto % de Nulos
                                    with blank_null_subtabs[1]:
                                        if len(blank_null_results['alto_nulo']) > 0:
                                            # Adicionar município aos dados, se disponível
                                            alto_nulo_df = blank_null_results['alto_nulo'].copy()
                                            if 'NM_MUNICIPIO' in df_filtered.columns:
                                                municipios = df_filtered[['NR_ZONA', 'NR_SECAO', 'NM_MUNICIPIO']].drop_duplicates()
                                                alto_nulo_df = pd.merge(alto_nulo_df, municipios, on=['NR_ZONA', 'NR_SECAO'], how='left')
                                            
                                            # Selecionar colunas para exibição
                                            display_cols = ['NM_MUNICIPIO', 'NR_ZONA', 'NR_SECAO', 'NULO', 'PCT_NULO', 'QT_VOTOS_TOTAL', 'QT_ELEITORES']
                                            display_cols = [col for col in display_cols if col in alto_nulo_df.columns]
                                            
                                            st.dataframe(
                                                alto_nulo_df[display_cols],
                                                use_container_width=True,
                                                column_config={
                                                    'PCT_NULO': st.column_config.NumberColumn(
                                                        '% Nulos',
                                                        help='Porcentagem de votos nulos',
                                                        format="%.2f%%"
                                                    ),
                                                    'NULO': st.column_config.NumberColumn(
                                                        'Votos Nulos',
                                                        help='Quantidade de votos nulos',
                                                        format="%d"
                                                    ),
                                                    'QT_VOTOS_TOTAL': st.column_config.NumberColumn(
                                                        'Total de Votos',
                                                        help='Total de votos na seção',
                                                        format="%d"
                                                    ),
                                                    'QT_ELEITORES': st.column_config.NumberColumn(
                                                        'Eleitores',
                                                        help='Total de eleitores na seção',
                                                        format="%d"
                                                    )
                                                }
                                            )
                                            
                                            # Download dos dados
                                            csv = alto_nulo_df.to_csv(index=False)
                                            st.download_button(
                                                label="Baixar dados de seções com alto % de nulos",
                                                data=csv,
                                                file_name="secoes_alto_percentual_nulos.csv",
                                                mime="text/csv"
                                            )
                                        else:
                                            st.info("Nenhuma seção com percentual anormalmente alto de votos nulos.")
                                    
                                    # Sub-tab 3: Alto % Combinado
                                    with blank_null_subtabs[2]:
                                        if len(blank_null_results['alto_branco_nulo']) > 0:
                                            # Adicionar município aos dados, se disponível
                                            alto_combinado_df = blank_null_results['alto_branco_nulo'].copy()
                                            if 'NM_MUNICIPIO' in df_filtered.columns:
                                                municipios = df_filtered[['NR_ZONA', 'NR_SECAO', 'NM_MUNICIPIO']].drop_duplicates()
                                                alto_combinado_df = pd.merge(alto_combinado_df, municipios, on=['NR_ZONA', 'NR_SECAO'], how='left')
                                            
                                            # Selecionar colunas para exibição
                                            display_cols = ['NM_MUNICIPIO', 'NR_ZONA', 'NR_SECAO', 'BRANCO', 'NULO', 'PCT_BRANCO_NULO', 'QT_VOTOS_TOTAL', 'QT_ELEITORES']
                                            display_cols = [col for col in display_cols if col in alto_combinado_df.columns]
                                            
                                            st.dataframe(
                                                alto_combinado_df[display_cols],
                                                use_container_width=True,
                                                column_config={
                                                    'PCT_BRANCO_NULO': st.column_config.NumberColumn(
                                                        '% Brancos+Nulos',
                                                        help='Porcentagem combinada de votos em branco e nulos',
                                                        format="%.2f%%"
                                                    ),
                                                    'BRANCO': st.column_config.NumberColumn(
                                                        'Votos Brancos',
                                                        help='Quantidade de votos em branco',
                                                        format="%d"
                                                    ),
                                                    'NULO': st.column_config.NumberColumn(
                                                        'Votos Nulos',
                                                        help='Quantidade de votos nulos',
                                                        format="%d"
                                                    ),
                                                    'QT_VOTOS_TOTAL': st.column_config.NumberColumn(
                                                        'Total de Votos',
                                                        help='Total de votos na seção',
                                                        format="%d"
                                                    ),
                                                    'QT_ELEITORES': st.column_config.NumberColumn(
                                                        'Eleitores',
                                                        help='Total de eleitores na seção',
                                                        format="%d"
                                                    )
                                                }
                                            )
                                            
                                            # Download dos dados
                                            csv = alto_combinado_df.to_csv(index=False)
                                            st.download_button(
                                                label="Baixar dados de seções com alto % combinado",
                                                data=csv,
                                                file_name="secoes_alto_percentual_combinado.csv",
                                                mime="text/csv"
                                            )
                                        else:
                                            st.info("Nenhuma seção com percentual anormalmente alto de votos em branco e nulos.")
                                    
                                    # Sub-tab 4: Visualizações aprimoradas
                                    with blank_null_subtabs[3]:
                                        todas_secoes = blank_null_results['todas_secoes']
                                        
                                        # Histograma da distribuição de votos brancos e nulos
                                        fig_hist = px.histogram(
                                            todas_secoes,
                                            x=['PCT_BRANCO', 'PCT_NULO'],
                                            nbins=50,
                                            opacity=0.7,
                                            barmode='overlay',
                                            title='Distribuição de Votos em Branco e Nulos',
                                            labels={'value': 'Porcentagem', 'variable': 'Tipo de Voto'},
                                            color_discrete_map={'PCT_BRANCO': '#2C3E50', 'PCT_NULO': '#E74C3C'}
                                        )
                                        
                                        # Adicionar linhas verticais com as médias
                                        fig_hist.add_vline(
                                            x=blank_null_results['media_branco'],
                                            line_dash="dash", line_color="#2C3E50",
                                            annotation_text=f"Média Brancos: {blank_null_results['media_branco']:.2f}%",
                                            annotation_position="top"
                                        )
                                        
                                        fig_hist.add_vline(
                                            x=blank_null_results['media_nulo'],
                                            line_dash="dash", line_color="#E74C3C",
                                            annotation_text=f"Média Nulos: {blank_null_results['media_nulo']:.2f}%",
                                            annotation_position="top"
                                        )
                                        
                                        # Adicionar linhas de limiar para detecção de anomalias (2.5 desvios padrão)
                                        fig_hist.add_vline(
                                            x=blank_null_results['media_branco'] + 2.5 * blank_null_results['std_branco'],
                                            line_dash="dot", line_color="#2C3E50",
                                            annotation_text=f"Limiar Brancos: {blank_null_results['media_branco'] + 2.5 * blank_null_results['std_branco']:.2f}%",
                                            annotation_position="bottom"
                                        )
                                        
                                        fig_hist.add_vline(
                                            x=blank_null_results['media_nulo'] + 2.5 * blank_null_results['std_nulo'],
                                            line_dash="dot", line_color="#E74C3C",
                                            annotation_text=f"Limiar Nulos: {blank_null_results['media_nulo'] + 2.5 * blank_null_results['std_nulo']:.2f}%",
                                            annotation_position="bottom"
                                        )
                                        
                                        st.plotly_chart(fig_hist, use_container_width=True)
                                        
                                        # Histograma da distribuição de votos combinados
                                        fig_hist_combined = px.histogram(
                                            todas_secoes,
                                            x='PCT_BRANCO_NULO',
                                            nbins=50,
                                            title='Distribuição do Percentual Combinado (Brancos + Nulos)',
                                            labels={'PCT_BRANCO_NULO': 'Porcentagem Combinada'},
                                            color_discrete_sequence=['#8E44AD']
                                        )
                                        
                                        # Adicionar linha da média e limiar
                                        fig_hist_combined.add_vline(
                                            x=blank_null_results['media_branco_nulo'],
                                            line_dash="dash", line_color="#8E44AD",
                                            annotation_text=f"Média: {blank_null_results['media_branco_nulo']:.2f}%",
                                            annotation_position="top"
                                        )
                                        
                                        fig_hist_combined.add_vline(
                                            x=blank_null_results['media_branco_nulo'] + 2.5 * blank_null_results['std_branco_nulo'],
                                            line_dash="dot", line_color="#8E44AD",
                                            annotation_text=f"Limiar: {blank_null_results['media_branco_nulo'] + 2.5 * blank_null_results['std_branco_nulo']:.2f}%",
                                            annotation_position="bottom"
                                        )
                                        
                                        st.plotly_chart(fig_hist_combined, use_container_width=True)
                                        
                                        # Scatter plot (brancos vs nulos)
                                        fig_scatter = px.scatter(
                                            todas_secoes,
                                            x='PCT_BRANCO',
                                            y='PCT_NULO',
                                            color='PCT_BRANCO_NULO',
                                            size='QT_VOTOS_TOTAL',
                                            hover_data=['NR_ZONA', 'NR_SECAO', 'QT_ELEITORES'],
                                            color_continuous_scale='Viridis',
                                            title='Relação entre Votos em Branco e Nulos por Seção',
                                            labels={
                                                'PCT_BRANCO': '% Votos em Branco',
                                                'PCT_NULO': '% Votos Nulos',
                                                'PCT_BRANCO_NULO': '% Total (Brancos + Nulos)',
                                                'QT_VOTOS_TOTAL': 'Total de Votos'
                                            }
                                        )
                                        
                                        # Adicionar linhas de referência
                                        fig_scatter.add_hline(
                                            y=blank_null_results['media_nulo'],
                                            line_dash="dash", line_color="gray",
                                            annotation_text=f"Média Nulos",
                                            annotation_position="left"
                                        )
                                        
                                        fig_scatter.add_vline(
                                            x=blank_null_results['media_branco'],
                                            line_dash="dash", line_color="gray",
                                            annotation_text=f"Média Brancos",
                                            annotation_position="top"
                                        )
                                        
                                        # Adicionar áreas de anomalia
                                        fig_scatter.add_hline(
                                            y=blank_null_results['media_nulo'] + 2.5 * blank_null_results['std_nulo'],
                                            line_dash="dot", line_color="red",
                                            annotation_text=f"Limiar Nulos",
                                            annotation_position="right"
                                        )
                                        
                                        fig_scatter.add_vline(
                                            x=blank_null_results['media_branco'] + 2.5 * blank_null_results['std_branco'],
                                            line_dash="dot", line_color="red",
                                            annotation_text=f"Limiar Brancos",
                                            annotation_position="bottom"
                                        )
                                        
                                        st.plotly_chart(fig_scatter, use_container_width=True)
                                        
                                        st.markdown("""
                                        #### Interpretação dos Gráficos
                                        
                                        - **Histogramas**: Mostram a distribuição dos percentuais de votos. A maioria das seções deve estar próxima da média, com poucos casos extremos.
                                        - **Scatter Plot**: Permite visualizar a relação entre votos em branco e nulos. Pontos além das linhas pontilhadas vermelhas são considerados anômalos.
                                        - **Gráfico Combinado**: Mostra o percentual total (brancos + nulos) em cada seção, destacando os casos acima do limiar de 2.5 desvios padrão.
                                        
                                        > **Dica**: Seções no canto superior direito do gráfico de dispersão são particularmente suspeitas, pois apresentam altos percentuais tanto de brancos quanto de nulos.
                                        """, help="Esta explicação ajuda a interpretar os gráficos e entender o significado estatístico dos limiares utilizados para detectar anomalias.")
                                        
                                    # Adicionar nova aba de comparativo
                                    with blank_null_subtabs[4]:
                                        st.subheader("Comparativo entre Regiões")
                                        
                                        # Verificar se temos informação de município ou UF
                                        geo_available = False
                                        geo_column = None
                                        
                                        if 'NM_MUNICIPIO' in df_filtered.columns:
                                            geo_column = 'NM_MUNICIPIO'
                                            geo_available = True
                                            geo_label = "Município"
                                        elif 'SG_UF' in df_filtered.columns:
                                            geo_column = 'SG_UF'
                                            geo_available = True
                                            geo_label = "UF"
                                        
                                        if geo_available:
                                            # Juntar informação geográfica com dados de brancos e nulos
                                            geo_data = df_filtered[[geo_column, 'NR_ZONA', 'NR_SECAO']].drop_duplicates()
                                            
                                            geo_blank_null = pd.merge(
                                                blank_null_results['todas_secoes'],
                                                geo_data,
                                                on=['NR_ZONA', 'NR_SECAO'],
                                                how='left'
                                            )
                                            
                                            # Calcular estatísticas por região
                                            regiao_stats = geo_blank_null.groupby(geo_column).agg({
                                                'PCT_BRANCO': ['mean', 'std', 'count'],
                                                'PCT_NULO': ['mean', 'std', 'count'],
                                                'PCT_BRANCO_NULO': ['mean', 'std']
                                            }).reset_index()
                                            
                                            # Arrumar nomes das colunas
                                            regiao_stats.columns = [
                                                geo_column if i == 0 else 
                                                f"{col}_{agg}" for i, (col, agg) in enumerate(regiao_stats.columns)
                                            ]
                                            
                                            # Ordenar por maior percentual combinado
                                            regiao_stats.sort_values('PCT_BRANCO_NULO_mean', ascending=False, inplace=True)
                                            
                                            # Limitar a um número razoável para visualização
                                            top_regioes = st.slider(
                                                f"Número de {geo_label}s para exibir", 
                                                min_value=5, 
                                                max_value=min(50, len(regiao_stats)), 
                                                value=20,
                                                help=f"Ajuste para ver mais ou menos {geo_label}s no gráfico comparativo"
                                            )
                                            
                                            regiao_stats_top = regiao_stats.head(top_regioes)
                                            
                                            # Criar gráfico comparativo
                                            fig_comp = px.bar(
                                                regiao_stats_top,
                                                x=geo_column,
                                                y=['PCT_BRANCO_mean', 'PCT_NULO_mean'],
                                                barmode='stack',
                                                error_y=[regiao_stats_top['PCT_BRANCO_std'], regiao_stats_top['PCT_NULO_std']],
                                                title=f'Comparativo de Votos em Branco e Nulos por {geo_label}',
                                                labels={
                                                    geo_column: geo_label,
                                                    'value': 'Percentual Médio (%)',
                                                    'variable': 'Tipo de Voto'
                                                },
                                                color_discrete_map={
                                                    'PCT_BRANCO_mean': '#2C3E50', 
                                                    'PCT_NULO_mean': '#E74C3C'
                                                },
                                                hover_data={
                                                    'PCT_BRANCO_std': True,
                                                    'PCT_NULO_std': True,
                                                    'PCT_BRANCO_count': True
                                                }
                                            )
                                            
                                            # Adicionar linha da média geral
                                            fig_comp.add_hline(
                                                y=blank_null_results['media_branco_nulo'],
                                                line_dash="dash", line_color="black",
                                                annotation_text=f"Média Geral: {blank_null_results['media_branco_nulo']:.2f}%",
                                                annotation_position="top right"
                                            )
                                            
                                            # Ajustar layout 
                                            fig_comp.update_layout(
                                                xaxis_tickangle=-45,
                                                yaxis_title="Percentual Médio (%)",
                                                legend_title="Tipo de Voto",
                                                hovermode="closest"
                                            )
                                            
                                            st.plotly_chart(fig_comp, use_container_width=True)
                                            
                                            # Tabela com dados
                                            st.subheader(f"Estatísticas por {geo_label}")
                                            
                                            # Renomear colunas para exibição
                                            display_cols = {
                                                geo_column: geo_label,
                                                'PCT_BRANCO_mean': 'Média % Brancos',
                                                'PCT_BRANCO_std': 'Desvio % Brancos',
                                                'PCT_NULO_mean': 'Média % Nulos',
                                                'PCT_NULO_std': 'Desvio % Nulos',
                                                'PCT_BRANCO_NULO_mean': 'Média % Combinado',
                                                'PCT_BRANCO_count': 'Nº Seções'
                                            }
                                            
                                            # Selecionar e renomear colunas
                                            table_data = regiao_stats[list(display_cols.keys())].copy()
                                            table_data.columns = list(display_cols.values())
                                            
                                            # Formatar números
                                            for col in table_data.columns:
                                                if col.startswith('Média') or col.startswith('Desvio'):
                                                    table_data[col] = table_data[col].round(2)
                                            
                                            st.dataframe(
                                                table_data,
                                                use_container_width=True,
                                                hide_index=True
                                            )
                                            
                                            # Opção para download
                                            csv_comp = table_data.to_csv(index=False)
                                            st.download_button(
                                                label=f"Baixar comparativo por {geo_label}",
                                                data=csv_comp,
                                                file_name=f"comparativo_brancos_nulos_por_{geo_label.lower()}.csv",
                                                mime="text/csv"
                                            )
                                        else:
                                            st.info(f"Não foi possível encontrar informações de município ou UF nos dados. "
                                                   f"Adicione estas colunas ao arquivo para habilitar o comparativo regional.")
                                        
                                        # Conclusão e explicação adicional
                                        st.markdown("""
                                        ### Interpretação dos Resultados
                                        
                                        - **Alto % de Brancos**: Pode indicar confusão dos eleitores ou escolha deliberada de não votar em nenhum candidato.
                                        - **Alto % de Nulos**: Pode indicar problemas com as urnas, eleitores que não sabem utilizar o sistema, 
                                          ou protestos conscientes.
                                        - **Alto % Combinado**: Uma porcentagem alta de ambos pode sinalizar problemas institucionais mais amplos ou 
                                          insatisfação generalizada.
                                        - **Comparativo Regional**: Diferenças significativas entre regiões podem indicar problemas específicos locais
                                          ou diferenças culturais e socioeconômicas.
                                        
                                        **Nota importante**: Esta análise destaca valores estatisticamente anômalos, que merecem investigação adicional,
                                        mas não são por si só evidência de fraude ou manipulação.
                                        """)

    # ABA 2: VISÃO GERAL (era a aba 1 antes)
    with tabs[1]:
        st.header("Visão Geral dos Dados")
        
        # Dashboard de Anomalias em Tempo Real
        st.subheader("📊 Dashboard de Anomalias em Tempo Real")
        
        anomalias_dashboard_col1, anomalias_dashboard_col2 = st.columns([2, 1])
        
        with anomalias_dashboard_col1:
            # Detectar anomalias usando método ensemble para mostrar no dashboard
            if 'QT_VOTOS' in df_filtered.columns:
                with st.spinner("Detectando anomalias para o dashboard..."):
                    # Aplicar métodos rápidos de detecção
                    df_dash = detect_anomalies_iqr(df_filtered.copy(), column='QT_VOTOS')
                    df_dash = detect_anomalies_zscore(df_dash, column='QT_VOTOS')
                    
                    # Marcar anomalias para o dashboard (combinação de métodos)
                    df_dash['ANOMALIA_DASHBOARD'] = (
                        df_dash['ANOMALIA_IQR'] | 
                        df_dash.get('ANOMALIA_ZSCORE', pd.Series(False, index=df_dash.index))
                    )
                    
                    # Ordenar por 'estranheza' (valores mais extremos)
                    if 'Z_SCORE' in df_dash.columns:
                        df_dash['SCORE_ANOMALIA'] = df_dash['Z_SCORE'].abs()
                        df_dash = df_dash.sort_values('SCORE_ANOMALIA', ascending=False)
                    
                    # Mostrar top anomalias
                    df_anomalias_top = df_dash[df_dash['ANOMALIA_DASHBOARD']]
                    
                    if len(df_anomalias_top) > 0:
                        cols_display = ['NM_MUNICIPIO', 'NR_ZONA', 'NR_SECAO', 'QT_VOTOS', 'QT_ELEITORES'] 
                        cols_display = [col for col in cols_display if col in df_anomalias_top.columns]
                        
                        st.dataframe(
                            df_anomalias_top[cols_display].head(10),
                            use_container_width=True,
                            column_config={
                                "QT_VOTOS": st.column_config.NumberColumn(
                                    "Votos",
                                    help="Quantidade de votos",
                                    format="%d",
                                    step=1,
                                ),
                                "QT_ELEITORES": st.column_config.NumberColumn(
                                    "Eleitores",
                                    help="Quantidade de eleitores",
                                    format="%d",
                                ),
                            }
                        )
                        
                        num_anomalias = len(df_anomalias_top)
                        pct_anomalias = (num_anomalias / len(df_filtered)) * 100
                        st.caption(f"Detectadas {num_anomalias} possíveis anomalias ({pct_anomalias:.2f}% do total)")
                    else:
                        st.info("Nenhuma anomalia detectada nos dados atuais.")
        
        with anomalias_dashboard_col2:
            # Métricas resumidas
            if 'QT_VOTOS' in df_filtered.columns:
                media_votos = df_filtered['QT_VOTOS'].mean()
                mediana_votos = df_filtered['QT_VOTOS'].median()
                max_votos = df_filtered['QT_VOTOS'].max()
                
                col_m1, col_m2 = st.columns(2)
                
                with col_m1:
                    st.metric("Média de Votos", f"{media_votos:.1f}")
                    st.metric("Máximo de Votos", f"{max_votos}")
                
                with col_m2:
                    st.metric("Mediana de Votos", f"{mediana_votos:.1f}")
                    
                    # Calcular coeficiente de Gini para desigualdade na distribuição de votos
                    # Quanto mais próximo de 1, maior a desigualdade
                    try:
                        votos_ordenados = df_filtered['QT_VOTOS'].sort_values()
                        n = len(votos_ordenados)
                        index = np.arange(1, n+1)
                        gini = (np.sum((2 * index - n - 1) * votos_ordenados)) / (n * np.sum(votos_ordenados))
                        st.metric("Índice de Gini", f"{gini:.3f}", help="Medida de desigualdade na distribuição (0-1)")
                    except:
                        pass
                
                # Indicador de Saúde dos Dados
                try:
                    # Calculando um score baseado em vários fatores
                    anomalias_detectadas = df_dash['ANOMALIA_DASHBOARD'].mean() * 100
                    skewness = abs(stats.skew(df_filtered['QT_VOTOS']))
                    missing_data = df_filtered['QT_VOTOS'].isna().mean() * 100
                    
                    # Score de 0 a 100, quanto maior, melhor a qualidade dos dados
                    data_health = 100 - (anomalias_detectadas * 0.5 + min(skewness * 10, 30) + missing_data)
                    data_health = max(0, min(100, data_health))
                    
                    # Cor baseada no score
                    if data_health >= 80:
                        color = "green"
                    elif data_health >= 60:
                        color = "orange"
                    else:
                        color = "red"
                    
                    st.markdown(f"""
                    <div style="margin-top: 1rem; padding: 1rem; border-radius: 0.5rem; background-color: #f0f2f6;">
                        <h4 style="margin: 0; margin-bottom: 0.5rem;">Saúde dos Dados</h4>
                        <div style="height: 1rem; width: 100%; background-color: #e1e4e8; border-radius: 0.5rem;">
                            <div style="height: 100%; width: {data_health}%; background-color: {color}; border-radius: 0.5rem;"></div>
                        </div>
                        <p style="margin-top: 0.5rem; text-align: center; font-weight: bold;">{data_health:.1f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                except:
                    pass
        
        # Linha divisória
        st.markdown("---")
        
        # Mostrar informações básicas
        st.subheader("Resumo do Dataset")
        st.write(f"Total de registros: {len(df_filtered):,}")
        st.write(f"Colunas disponíveis: {len(df_filtered.columns)}")
        
        # Mostrar primeiras linhas
        st.subheader("Amostra dos Dados")
        st.dataframe(df_filtered.head(10))
        
        # Estatísticas descritivas
        st.subheader("Estatísticas Descritivas")
        st.dataframe(df_filtered.describe())
        
        # Visualização de distribuição por município (top 20)
        if 'NM_MUNICIPIO' in df_filtered.columns and col_votos in df_filtered.columns:
            st.subheader("Distribuição de Votos por Município (Top 20)")
            
            # Agregar por município
            municipio_votes = df_filtered.groupby('NM_MUNICIPIO')[col_votos].sum().sort_values(ascending=False).head(20)
            
            # Criar gráfico de barras
            fig = px.bar(
                x=municipio_votes.index,
                y=municipio_votes.values,
                labels={'x': 'Município', 'y': 'Total de Votos'},
                title='Top 20 Municípios por Total de Votos'
            )
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig)
    
    # NOVA ABA 2: ANÁLISE POR CANDIDATO
    with tabs[2]:
        st.header("Análise por Candidato")
        
        # Verificar se temos as colunas necessárias
        if 'NM_VOTAVEL' in df_filtered.columns:
            # Listar candidatos disponíveis
            candidatos = sorted(df_filtered['NM_VOTAVEL'].dropna().unique())
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                # Seleção de candidato para análise
                candidato_selecionado = st.selectbox(
                    "Selecione um candidato para análise detalhada",
                    candidatos
                )
                
                # Mostrar informações básicas do candidato
                df_candidato = df_filtered[df_filtered['NM_VOTAVEL'] == candidato_selecionado]
                
                # Estatísticas básicas
                total_votos = df_candidato[col_votos].sum()
                total_secoes = df_candidato.groupby(['NR_ZONA', 'NR_SECAO']).ngroups
                media_votos_secao = total_votos / total_secoes if total_secoes > 0 else 0
                
                st.metric("Total de Votos", f"{total_votos:,}")
                st.metric("Total de Seções", f"{total_secoes:,}")
                st.metric("Média de Votos por Seção", f"{media_votos_secao:.1f}")
                
                # Opção para análise de concentração de votos
                st.subheader("Análise de Concentração")
                threshold = st.slider(
                    "Limiar de dominância (%)", 
                    min_value=50, 
                    max_value=100, 
                    value=95,
                    step=5,
                    help="Mostrar seções onde o candidato teve pelo menos esta % dos votos."
                )
                
                min_votos = st.number_input(
                    "Mínimo de votos na seção", 
                    min_value=1, 
                    value=50,
                    help="Filtrar apenas seções com pelo menos este número total de votos."
                )
            
            with col2:
                # Mostrar distribuição de votos por seção para o candidato
                st.subheader(f"Distribuição de Votos de {candidato_selecionado}")
                
                # Agrupar por zona e seção
                df_votos_secao = df_candidato.groupby(['NR_ZONA', 'NR_SECAO'])[col_votos].sum().reset_index()
                
                # Plotar histograma
                fig = px.histogram(
                    df_votos_secao, 
                    x=col_votos,
                    nbins=50,
                    title=f'Distribuição de Votos por Seção - {candidato_selecionado}',
                    labels={col_votos: 'Quantidade de Votos'}
                )
                fig.update_layout(bargap=0.1)
                st.plotly_chart(fig)
                
                # Analisar dominância do candidato nas seções
                st.subheader("Análise de Dominância por Seção")
                
                with st.spinner("Analisando dominância nas seções..."):
                    # Preparar dados para análise de dominância
                    dominance_df, error = analyze_candidate_dominance(
                        df_filtered, 
                        candidate_col='NM_VOTAVEL', 
                        vote_col=col_votos
                    )
                    
                    if dominance_df is not None:
                        # Filtrar para o candidato selecionado
                        dom_candidato = dominance_df[
                            (dominance_df['NM_VOTAVEL'] == candidato_selecionado) & 
                            (dominance_df['TOTAL_VOTOS_SECAO'] >= min_votos) &
                            (dominance_df['PERCENTUAL_DOMINANCIA'] >= threshold)
                        ]
                        
                        if len(dom_candidato) > 0:
                            # Mostrar tabela com seções onde o candidato domina
                            st.write(f"**Seções onde {candidato_selecionado} tem pelo menos {threshold}% dos votos:**")
                            
                            display_cols = [
                                'NR_ZONA', 'NR_SECAO', 'VOTOS_CANDIDATO_DOMINANTE', 
                                'TOTAL_VOTOS_SECAO', 'PERCENTUAL_DOMINANCIA', 
                                'NUM_CANDIDATOS', 'DOMINANCIA_EXTREMA'
                            ]
                            
                            st.dataframe(
                                dom_candidato[display_cols],
                                column_config={
                                    'PERCENTUAL_DOMINANCIA': st.column_config.ProgressColumn(
                                        '% dos Votos',
                                        help='Percentual dos votos da seção recebidos pelo candidato',
                                        format="%.2f%%",
                                        min_value=0,
                                        max_value=100,
                                    ),
                                    'DOMINANCIA_EXTREMA': st.column_config.CheckboxColumn(
                                        'Dominância Extrema',
                                        help='Indica se o candidato recebeu praticamente 100% dos votos'
                                    )
                                }
                            )
                            
                            # Baixar dados
                            csv = dom_candidato.to_csv(index=False)
                            st.download_button(
                                label="Baixar dados de dominância como CSV",
                                data=csv,
                                file_name=f"dominancia_{candidato_selecionado.replace(' ', '_')}.csv",
                                mime="text/csv"
                            )
                            
                            # Analisar distribuição geográfica da dominância
                            if 'NM_MUNICIPIO' in df_filtered.columns:
                                # Mapa de calor por município
                                st.subheader("Concentração de Dominância por Município")
                                
                                # Juntar com os dados originais para obter município
                                municipios_info = df_filtered[['NR_ZONA', 'NR_SECAO', 'NM_MUNICIPIO', 'SG_UF']].drop_duplicates()
                                dom_municipios = pd.merge(dom_candidato, municipios_info, on=['NR_ZONA', 'NR_SECAO'])
                                
                                # Agrupar por município
                                mapa_dominancia = dom_municipios.groupby('NM_MUNICIPIO').agg({
                                    'NR_ZONA': 'count',  # Contador de seções
                                    'PERCENTUAL_DOMINANCIA': 'mean',  # Média de dominância
                                    'DOMINANCIA_EXTREMA': 'sum'  # Contagem de dominância extrema
                                }).reset_index()
                                
                                mapa_dominancia.columns = ['MUNICIPIO', 'NUM_SECOES', 'MEDIA_DOMINANCIA', 'SECOES_EXTREMAS']
                                
                                # Ordenar por número de seções
                                mapa_dominancia = mapa_dominancia.sort_values('NUM_SECOES', ascending=False)
                                
                                # Mostrar mapa de calor ou gráfico de barras
                                fig = px.bar(
                                    mapa_dominancia.head(20), 
                                    x='MUNICIPIO', 
                                    y='NUM_SECOES',
                                    color='MEDIA_DOMINANCIA',
                                    text='SECOES_EXTREMAS',
                                    color_continuous_scale="Reds",
                                    title=f'Municípios com Maior Dominância de {candidato_selecionado}',
                                    labels={
                                        'MUNICIPIO': 'Município', 
                                        'NUM_SECOES': 'Número de Seções', 
                                        'MEDIA_DOMINANCIA': '% Média de Dominância',
                                        'SECOES_EXTREMAS': 'Seções com Dominância Extrema'
                                    }
                                )
                                fig.update_layout(xaxis_tickangle=-45)
                                st.plotly_chart(fig)
                        else:
                            st.info(f"Não foram encontradas seções onde {candidato_selecionado} tenha pelo menos {threshold}% dos votos.")
                    else:
                        st.error(f"Erro na análise de dominância: {error}")
                
                # Comparativo com outros candidatos
                st.subheader("Comparativo com Outros Candidatos")
                
                # Selecionar candidatos para comparação
                outros_candidatos = st.multiselect(
                    "Selecione candidatos para comparar",
                    [c for c in candidatos if c != candidato_selecionado],
                    max_selections=3
                )
                
                if outros_candidatos:
                    # Agrupar dados para comparação
                    candidatos_comparar = [candidato_selecionado] + outros_candidatos
                    df_comparacao = df_filtered[df_filtered['NM_VOTAVEL'].isin(candidatos_comparar)]
                    
                    # Agrupar por candidato
                    comparacao = df_comparacao.groupby('NM_VOTAVEL').agg({
                        col_votos: 'sum',
                        'NR_ZONA': lambda x: len(x.unique())
                    }).reset_index()
                    
                    comparacao.columns = ['CANDIDATO', 'TOTAL_VOTOS', 'ZONAS_PRESENTES']
                    
                    # Gráfico de comparação
                    fig = px.bar(
                        comparacao,
                        x='CANDIDATO',
                        y='TOTAL_VOTOS',
                        color='CANDIDATO',
                        title='Comparação de Votos Entre Candidatos',
                        text='TOTAL_VOTOS',
                        labels={'TOTAL_VOTOS': 'Total de Votos', 'CANDIDATO': 'Candidato'}
                    )
                    fig.update_traces(texttemplate='%{text:,}', textposition='outside')
                    st.plotly_chart(fig)
        else:
            st.error("Coluna 'NM_VOTAVEL' não encontrada nos dados. A análise por candidato não está disponível.")
    
    # ABA 3: DETECÇÃO DE ANOMALIAS (era a aba 2 antes)
    with tabs[3]:
        st.header("Detecção de Anomalias")
        
        # Método de detecção
        col1, col2 = st.columns(2)
        
        with col1:
            metodo_anomalia = st.selectbox(
                "Selecione o método de detecção:",
                ["IQR", "Z-Score", "Isolation Forest", "DBSCAN", "GMM", "PCA", "Autoencoder", "Ensemble"]
            )
        
        with col2:
            if metodo_anomalia != "Ensemble":
                threshold = st.slider(
                    "Threshold de detecção (percentil/sensibilidade)", 
                    min_value=0.1, 
                    max_value=10.0, 
                    value=1.0, 
                    step=0.1
                )
            else:
                min_métodos = st.slider(
                    "Mínimo de métodos para considerar anomalia", 
                    min_value=1, 
                    max_value=5, 
                    value=2
                )
        
        # Aplicar o método selecionado
        df_anomalias = df_filtered.copy()
        
        # Aplicação do método escolhido
        if metodo_anomalia == "IQR":
            df_anomalias = detect_anomalies_iqr(df_anomalias, column=col_votos)
            num_anomalias = df_anomalias['ANOMALIA_IQR'].sum()
            anomalia_col = 'ANOMALIA_IQR'
            
        elif metodo_anomalia == "Z-Score":
            df_anomalias = detect_anomalies_zscore(df_anomalias, column=col_votos, z_thresh=threshold)
            if 'ANOMALIA_ZSCORE' in df_anomalias.columns:
                num_anomalias = df_anomalias['ANOMALIA_ZSCORE'].sum()
                anomalia_col = 'ANOMALIA_ZSCORE'
            else:
                st.warning("Não foi possível calcular Z-Score (desvio padrão zero).")
                num_anomalias = 0
                anomalia_col = None
                
        elif metodo_anomalia == "Isolation Forest":
            df_anomalias = detect_anomalies_isolation_forest(df_anomalias, columns=[col_votos])
            num_anomalias = df_anomalias['ANOMALIA_IF'].sum()
            anomalia_col = 'ANOMALIA_IF'
            
        elif metodo_anomalia == "DBSCAN":
            df_anomalias = detect_anomalies_dbscan(df_anomalias, columns=[col_votos])
            num_anomalias = df_anomalias['ANOMALIA_DBSCAN'].sum()
            anomalia_col = 'ANOMALIA_DBSCAN'
            
        elif metodo_anomalia == "GMM":
            df_anomalias = detect_anomalies_gmm(df_anomalias, columns=[col_votos], threshold=threshold/100)
            num_anomalias = df_anomalias['ANOMALIA_GMM'].sum()
            anomalia_col = 'ANOMALIA_GMM'
            
        elif metodo_anomalia == "PCA":
            columns_for_pca = [col for col in [col_votos, 'QT_ELEITORES'] if col in df_anomalias.columns]
            df_anomalias = analyze_multivariate_anomalies(df_anomalias, columns=columns_for_pca)
            num_anomalias = df_anomalias['ANOMALIA_MAHALANOBIS'].sum()
            anomalia_col = 'ANOMALIA_MAHALANOBIS'
            
        elif metodo_anomalia == "Autoencoder":
            with st.spinner("Aplicando Autoencoder (pode demorar um pouco)..."):
                columns_for_ae = [col for col in [col_votos, 'QT_ELEITORES'] if col in df_anomalias.columns]
                df_anomalias = apply_autoencoder(df_anomalias, columns=columns_for_ae, contamination=threshold/100)
            
            if 'ANOMALIA_AUTOENCODER' in df_anomalias.columns:
                num_anomalias = df_anomalias['ANOMALIA_AUTOENCODER'].sum()
                anomalia_col = 'ANOMALIA_AUTOENCODER'
            else:
                st.error("Falha ao aplicar autoencoder. Verifique se o TensorFlow está instalado.")
                num_anomalias = 0
                anomalia_col = None
                
        elif metodo_anomalia == "Ensemble":
            with st.spinner("Aplicando métodos de ensemble..."):
                # Aplicar todos os métodos
                df_temp = detect_anomalies_iqr(df_anomalias, column=col_votos)
                df_temp = detect_anomalies_zscore(df_temp, column=col_votos)
                df_temp = detect_anomalies_isolation_forest(df_temp, columns=[col_votos])
                df_temp = detect_anomalies_dbscan(df_temp, columns=[col_votos])
                df_temp = detect_anomalies_gmm(df_temp, columns=[col_votos])
                
                # Contar quantos métodos consideram cada registro como anomalia
                df_temp['VOTOS_ANOMALIA'] = 0
                
                if 'ANOMALIA_IQR' in df_temp.columns:
                    df_temp['VOTOS_ANOMALIA'] += df_temp['ANOMALIA_IQR'].astype(int)
                if 'ANOMALIA_ZSCORE' in df_temp.columns:
                    df_temp['VOTOS_ANOMALIA'] += df_temp['ANOMALIA_ZSCORE'].astype(int)
                if 'ANOMALIA_IF' in df_temp.columns:
                    df_temp['VOTOS_ANOMALIA'] += df_temp['ANOMALIA_IF'].astype(int)
                if 'ANOMALIA_DBSCAN' in df_temp.columns:
                    df_temp['VOTOS_ANOMALIA'] += df_temp['ANOMALIA_DBSCAN'].astype(int)
                if 'ANOMALIA_GMM' in df_temp.columns:
                    df_temp['VOTOS_ANOMALIA'] += df_temp['ANOMALIA_GMM'].astype(int)
                
                # Marcar como anomalia se pelo menos min_métodos métodos considerarem como tal
                df_temp['ANOMALIA_ENSEMBLE'] = df_temp['VOTOS_ANOMALIA'] >= min_métodos
                
                # Atualizar DataFrame
                df_anomalias = df_temp.copy()
                num_anomalias = df_anomalias['ANOMALIA_ENSEMBLE'].sum()
                anomalia_col = 'ANOMALIA_ENSEMBLE'
        
        # Calcular pontuação de anomalias
        if anomalia_col and num_anomalias > 0:
            with st.spinner("Calculando pontuação de anomalias..."):
                df_anomalias = calculate_anomaly_score(df_anomalias)
        
        # Exibir resultados
        col_metrics1, col_metrics2, col_metrics3 = st.columns(3)
        
        with col_metrics1:
            st.metric("Anomalias Detectadas", f"{num_anomalias}")
            st.caption(f"({(num_anomalias / len(df_anomalias)) * 100:.2f}% dos registros)")
        
        # Mostrar métricas adicionais se tivermos anomalias
        if anomalia_col and num_anomalias > 0 and 'SCORE_ANOMALIA' in df_anomalias.columns:
            with col_metrics2:
                score_medio = df_anomalias[df_anomalias[anomalia_col]]['SCORE_ANOMALIA'].mean()
                st.metric("Score Médio de Anomalias", f"{score_medio:.1f}/100")
                
                # Contar anomalias por severidade
                if 'SEVERIDADE' in df_anomalias.columns:
                    severidade_counts = df_anomalias[df_anomalias[anomalia_col]]['SEVERIDADE'].value_counts()
                    
                    # Verificar se há anomalias de alta severidade
                    alta_severidade = (
                        severidade_counts.get('Muito Alta', 0) + 
                        severidade_counts.get('Alta', 0)
                    )
                    
                    st.caption(f"Anomalias de alta severidade: {alta_severidade}")
            
            with col_metrics3:
                # Mostrar distribuição de severidade em um pequeno gráfico
                if 'SEVERIDADE' in df_anomalias.columns:
                    sev_data = df_anomalias[df_anomalias[anomalia_col]]['SEVERIDADE'].value_counts().reset_index()
                    sev_data.columns = ['Severidade', 'Contagem']
                    
                    # Ordenar por severidade
                    severidade_ordem = ['Muito Baixa', 'Baixa', 'Média', 'Alta', 'Muito Alta']
                    sev_data['Severidade'] = pd.Categorical(
                        sev_data['Severidade'], 
                        categories=severidade_ordem, 
                        ordered=True
                    )
                    sev_data = sev_data.sort_values('Severidade')
                    
                    # Cores por severidade
                    cores_severidade = {
                        'Muito Baixa': 'green',
                        'Baixa': 'lightgreen',
                        'Média': 'yellow',
                        'Alta': 'orange',
                        'Muito Alta': 'red'
                    }
                    
                    # Gráfico de barras
                    fig = px.bar(
                        sev_data, 
                        x='Severidade', 
                        y='Contagem',
                        color='Severidade',
                        color_discrete_map=cores_severidade,
                        title='Distribuição de Severidade'
                    )
                    fig.update_layout(height=200)
                    st.plotly_chart(fig, use_container_width=True)
        
        # Se tivermos anomalias, mostrar visualizações
        if anomalia_col and num_anomalias > 0:
            # Criar abas para diferentes visualizações
            anomaly_viz_tabs = st.tabs(["Ranking de Anomalias", "Gráficos", "Mapa Geoespacial", "Tabela Completa"])
            
            # Aba 1: Ranking de Anomalias
            with anomaly_viz_tabs[0]:
                st.subheader("Ranking de Anomalias por Severidade")
                
                if 'SCORE_ANOMALIA' in df_anomalias.columns:
                    # Filtrar apenas anomalias e ordenar por score
                    df_anomalias_ranking = df_anomalias[df_anomalias[anomalia_col]].sort_values('SCORE_ANOMALIA', ascending=False)
                    
                    # Preparar colunas para exibição
                    display_cols = ['NM_MUNICIPIO', 'NR_ZONA', 'NR_SECAO', col_votos, 'QT_ELEITORES', 'SCORE_ANOMALIA', 'SEVERIDADE']
                    display_cols = [col for col in display_cols if col in df_anomalias_ranking.columns]
                    
                    # Mostrar as top 20 anomalias mais severas
                    st.dataframe(
                        df_anomalias_ranking[display_cols].head(20),
                        column_config={
                            'SCORE_ANOMALIA': st.column_config.ProgressColumn(
                                'Score de Anomalia',
                                help='Pontuação de 0-100 indicando severidade da anomalia',
                                format="%d",
                                min_value=0,
                                max_value=100,
                            ),
                            'SEVERIDADE': st.column_config.Column(
                                'Severidade',
                                help='Classificação da severidade da anomalia',
                                width='medium',
                            ),
                            col_votos: st.column_config.NumberColumn(
                                'Votos',
                                help='Quantidade de votos',
                                format="%d",
                            ),
                            'QT_ELEITORES': st.column_config.NumberColumn(
                                'Eleitores',
                                help='Quantidade de eleitores',
                                format="%d",
                            ),
                        }
                    )
                else:
                    st.info("Pontuação de anomalias não disponível para este método.")
            
            # Aba 2: Gráficos de Dispersão
            with anomaly_viz_tabs[1]:
                # Gráfico de dispersão para 'QT_ELEITORES' x 'QT_VOTOS' se disponíveis
                if all(col in df_anomalias.columns for col in ['QT_ELEITORES', col_votos]):
                    st.subheader("Dispersão de Anomalias")
                    
                    # Se temos pontuação de anomalias, usar como tamanho dos pontos
                    if 'SCORE_ANOMALIA' in df_anomalias.columns and 'SEVERIDADE' in df_anomalias.columns:
                        fig = px.scatter(
                            df_anomalias, 
                            x='QT_ELEITORES', 
                            y=col_votos,
                            color=anomalia_col,
                            size='SCORE_ANOMALIA',
                            color_discrete_map={True: 'red', False: 'blue'},
                            hover_data=['NM_MUNICIPIO', 'NR_ZONA', 'NR_SECAO', 'SEVERIDADE', 'SCORE_ANOMALIA'] 
                            if 'NM_MUNICIPIO' in df_anomalias.columns else None,
                            title=f'Dispersão de Anomalias por {metodo_anomalia}',
                            labels={
                                'QT_ELEITORES': 'Quantidade de Eleitores', 
                                col_votos: 'Quantidade de Votos',
                                anomalia_col: 'É Anomalia',
                                'SCORE_ANOMALIA': 'Score de Anomalia',
                                'SEVERIDADE': 'Severidade'
                            }
                        )
                    else:
                        fig = px.scatter(
                            df_anomalias, 
                            x='QT_ELEITORES', 
                            y=col_votos,
                            color=anomalia_col,
                            color_discrete_map={True: 'red', False: 'blue'},
                            hover_data=['NM_MUNICIPIO', 'NR_ZONA', 'NR_SECAO'] 
                            if 'NM_MUNICIPIO' in df_anomalias.columns else None,
                            title=f'Dispersão de Anomalias por {metodo_anomalia}',
                            labels={'QT_ELEITORES': 'Quantidade de Eleitores', col_votos: 'Quantidade de Votos'}
                        )
                    st.plotly_chart(fig)
            
            # Aba 3: Mapa Geoespacial
            with anomaly_viz_tabs[2]:
                st.subheader("Distribuição Geoespacial de Anomalias")
                
                try:
                    import streamlit_folium
                    
                    with st.spinner("Gerando mapa geoespacial..."):
                        # Criar mapa geoespacial
                        mapa, erro = create_geo_map(df_anomalias, anomaly_col=anomalia_col)
                        
                        if mapa:
                            st.success(f"Mapa gerado com {num_anomalias} anomalias")
                            streamlit_folium.folium_static(mapa)
                        else:
                            st.warning(f"Não foi possível gerar o mapa: {erro}")
                            
                            # Alternativa: mostrar mapa de calor simplificado usando Plotly
                            if 'SG_UF' in df_anomalias.columns:
                                st.subheader("Mapa de Calor por UF (Alternativa)")
                                
                                # Agrupar anomalias por UF
                                anomalias_por_uf = df_anomalias.groupby('SG_UF')[anomalia_col].sum().reset_index()
                                anomalias_por_uf.columns = ['SG_UF', 'NUM_ANOMALIAS']
                                
                                # Criar mapa de calor
                                fig = px.choropleth(
                                    anomalias_por_uf,
                                    locations='SG_UF',
                                    color='NUM_ANOMALIAS',
                                    scope="south america",
                                    title="Densidade de Anomalias por UF",
                                    color_continuous_scale='Reds',
                                    labels={'NUM_ANOMALIAS': 'Anomalias Detectadas'}
                                )
                                fig.update_geos(fitbounds="locations", visible=False)
                                st.plotly_chart(fig)
                except ImportError:
                    st.error("Biblioteca streamlit-folium não está instalada. Instale com: pip install streamlit-folium")
            
            # Aba 4: Tabela Completa
            with anomaly_viz_tabs[3]:
                # Tabela com as anomalias
                st.subheader("Tabela de Anomalias Detectadas")
                
                df_outliers = df_anomalias[df_anomalias[anomalia_col] == True]
                
                # Colunas a mostrar
                cols_to_show = [col for col in ['NM_MUNICIPIO', 'NR_ZONA', 'NR_SECAO', 'SG_UF', col_votos, 'QT_ELEITORES', 'SCORE_ANOMALIA', 'SEVERIDADE'] 
                               if col in df_outliers.columns]
                st.dataframe(df_outliers[cols_to_show])
                
                # Download de anomalias
                csv = df_outliers.to_csv(index=False)
                st.download_button(
                    label="Baixar anomalias como CSV",
                    data=csv,
                    file_name=f"anomalias_{metodo_anomalia}.csv",
                    mime="text/csv"
                )
        else:
            st.info("Nenhuma anomalia detectada para visualizar.")
    
    # ABA 4: ANÁLISE ESTATÍSTICA
    with tabs[4]:
        st.header("Análise Estatística")
        
        # Análise da distribuição dos dados
        st.subheader("Distribuição dos Votos")
        
        # Histograma
        fig = px.histogram(
            df_filtered, 
            x=col_votos,
            nbins=50,
            title='Distribuição dos Votos',
            labels={col_votos: 'Quantidade de Votos'}
        )
        fig.update_layout(bargap=0.1)
        st.plotly_chart(fig)
        
        # QQ-Plot para testar normalidade
        st.subheader("QQ-Plot (Teste de Normalidade)")
        qqplot_data = stats.probplot(df_filtered[col_votos].dropna(), dist="norm")
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.array([qqplot_data[0][0][0], qqplot_data[0][0][-1]])
        ax.plot(qqplot_data[0][0], qqplot_data[0][1], 'o', markersize=5)
        ax.plot(x, qqplot_data[1][0] + qqplot_data[1][1]*x, 'r-', linewidth=2)
        ax.set_title('QQ-Plot de Votos')
        ax.set_xlabel('Quantis Teóricos')
        ax.set_ylabel('Quantis Ordenados')
        st.pyplot(fig)
        
        # Testes estatísticos
        st.subheader("Testes Estatísticos")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Teste de Shapiro-Wilk para normalidade
            sample = df_filtered[col_votos].dropna().sample(min(5000, len(df_filtered)))
            shapiro_stat, shapiro_p = stats.shapiro(sample)
            st.write("**Teste de Shapiro-Wilk (Normalidade)**")
            st.write(f"Estatística: {shapiro_stat:.4f}")
            st.write(f"Valor p: {shapiro_p:.8f}")
            st.write(f"Conclusão: {'Normal' if shapiro_p > 0.05 else 'Não-Normal'}")
        
        with col2:
            # Teste D'Agostino-Pearson
            k2, dagostino_p = stats.normaltest(sample)
            st.write("**Teste D'Agostino-Pearson (Normalidade)**")
            st.write(f"Estatística: {k2:.4f}")
            st.write(f"Valor p: {dagostino_p:.8f}")
            st.write(f"Conclusão: {'Normal' if dagostino_p > 0.05 else 'Não-Normal'}")
        
        # Lei de Benford
        st.subheader("Análise da Lei de Benford")
        benford_df, benford_stats = analyze_benford_law(df_filtered)
        
        # Gráfico da Lei de Benford
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Barras para frequência observada
        fig.add_trace(
            go.Bar(
                x=benford_df['DIGITO'], 
                y=benford_df['FREQ_OBSERVADA'],
                name="Frequência Observada",
                marker_color='royalblue'
            )
        )
        
        # Linha para frequência esperada
        fig.add_trace(
            go.Scatter(
                x=benford_df['DIGITO'], 
                y=benford_df['FREQ_ESPERADA'],
                name="Frequência Esperada (Benford)",
                marker_color='red',
                mode='lines+markers'
            )
        )
        
        # Layout
        fig.update_layout(
            title='Análise da Lei de Benford - Primeiro Dígito',
            xaxis_title='Primeiro Dígito',
            yaxis_title='Frequência',
            barmode='group',
            xaxis = dict(
                tickmode = 'array',
                tickvals = benford_df['DIGITO']
            )
        )
        
        st.plotly_chart(fig)
        
        # Exibir estatísticas do teste
        st.write(f"**Estatística Chi²:** {benford_stats['chi2']:.4f}")
        st.write(f"**Valor p:** {benford_stats['p_value']:.8f}")
        st.write(f"**Conformidade com Lei de Benford:** {'Sim' if benford_stats['benford_compliant'] else 'Não'}")
        
        if not benford_stats['benford_compliant']:
            st.warning("⚠️ Os dados **não seguem** a Lei de Benford, o que pode indicar possíveis anomalias ou manipulação.")
        else:
            st.success("✅ Os dados seguem a Lei de Benford, o que é esperado para dados naturais.")

    # ABA 5: ANÁLISE TEMPORAL (Se houver dados temporais)
    with tabs[5]:
        st.header("Análise Temporal")
        
        # Verificar se existe coluna de data
        date_columns = [col for col in df_filtered.columns if 'DATA' in col.upper() or 'DT_' in col.upper()]
        
        if date_columns:
            # Selecionar coluna de data
            date_col = st.selectbox("Selecione a coluna de data", date_columns)
            
            # Selecionar coluna de valor
            value_col = st.selectbox("Selecione a coluna de valor", 
                                    df_filtered.select_dtypes(include=['number']).columns.tolist())
            
            # Selecionar coluna de agrupamento opcional
            group_cols = [None] + [col for col in df_filtered.columns if col not in [date_col, value_col]]
            group_col = st.selectbox("Selecione uma coluna para agrupar (opcional)", group_cols)
            
            # Realizar análise temporal
            with st.spinner("Gerando análise temporal..."):
                if date_col and value_col:
                    try:
                        fig, ts_data = temporal_analysis(df_filtered, date_col, value_col, group_col)
                        st.plotly_chart(fig)
                        
                        # Exibir dados da série temporal
                        st.subheader("Dados da Série Temporal")
                        st.dataframe(ts_data.head(20))
                        
                        # Download dos dados temporais
                        csv = ts_data.to_csv(index=False)
                        st.download_button(
                            label="Baixar dados temporais como CSV",
                            data=csv,
                            file_name="dados_temporais.csv",
                            mime="text/csv"
                        )
                    except Exception as e:
                        st.error(f"Erro ao gerar análise temporal: {e}")
        else:
            st.warning("Não foram encontradas colunas de data no conjunto de dados.")
    
    # ABA 6: MACHINE LEARNING
    with tabs[6]:
        st.header("Análise com Machine Learning")
        
        # Selecionar tipo de modelo
        model_type = st.selectbox(
            "Selecione o tipo de modelo",
            ["Isolation Forest", "XGBoost"]
        )
        
        # Selecionar features para o modelo
        numeric_cols = df_filtered.select_dtypes(include=['number']).columns.tolist()
        selected_features = st.multiselect(
            "Selecione as features para o modelo",
            numeric_cols,
            default=[col for col in numeric_cols if col in ['QT_VOTOS', 'QT_ELEITORES']][:2]
        )
        
        if selected_features:
            if model_type == "XGBoost":
                target_col = st.selectbox(
                    "Selecione a coluna alvo (para regressão/classificação)",
                    [col for col in numeric_cols if col not in selected_features]
                )
            else:
                target_col = None
            
            # Gerar explicações do modelo
            with st.spinner("Treinando modelo e gerando explicações..."):
                try:
                    fig_dict, most_important_feature = model_explanation(
                        df_filtered, 
                        model_type.lower(), 
                        selected_features,
                        target_col
                    )
                    
                    if fig_dict:
                        st.subheader("Importância das Features")
                        st.pyplot(fig_dict['importance'])
                        
                        st.subheader(f"Dependência da Feature Mais Importante: {most_important_feature}")
                        st.pyplot(fig_dict['dependence'])
                        
                        st.success(f"A feature mais importante para o modelo é: {most_important_feature}")
                    else:
                        st.error("Não foi possível gerar explicações. Verifique se SHAP está instalado.")
                except Exception as e:
                    st.error(f"Erro ao treinar modelo: {e}")
        else:
            st.warning("Selecione pelo menos uma feature para o modelo.")
    
    # ABA 7: EXPORTAÇÃO
    with tabs[7]:
        st.header("Exportação de Dados e Relatórios")
        
        # Opções de exportação
        st.subheader("Exportar Dados Filtrados")
        export_format = st.radio(
            "Selecione o formato de exportação",
            ["CSV", "Excel", "JSON", "Parquet"]
        )
        
        # Botão para exportar
        if st.button("Gerar arquivo para download"):
            with st.spinner("Preparando arquivo para download..."):
                if export_format == "CSV":
                    file_data = df_filtered.to_csv(index=False)
                    file_name = "dados_eleicoes_filtrados.csv"
                    mime = "text/csv"
                elif export_format == "Excel":
                    import io
                    buffer = io.BytesIO()
                    df_filtered.to_excel(buffer, index=False)
                    file_data = buffer.getvalue()
                    file_name = "dados_eleicoes_filtrados.xlsx"
                    mime = "application/vnd.ms-excel"
                elif export_format == "JSON":
                    file_data = df_filtered.to_json(orient='records')
                    file_name = "dados_eleicoes_filtrados.json"
                    mime = "application/json"
                elif export_format == "Parquet":
                    import io
                    buffer = io.BytesIO()
                    df_filtered.to_parquet(buffer, index=False)
                    file_data = buffer.getvalue()
                    file_name = "dados_eleicoes_filtrados.parquet"
                    mime = "application/octet-stream"
                
                st.download_button(
                    label=f"Baixar dados como {export_format}",
                    data=file_data,
                    file_name=file_name,
                    mime=mime
                )
        
        # Exportar resumo estatístico
        st.subheader("Exportar Resumo Estatístico")
        if st.button("Gerar resumo estatístico"):
            with st.spinner("Preparando resumo estatístico..."):
                # Estatísticas descritivas
                desc_stats = df_filtered.describe().reset_index()
                
                # Testes estatísticos
                test_results = perform_statistical_tests(df_filtered)
                
                # Análise de Benford
                benford_results, benford_stats = analyze_benford_law(df_filtered)
                
                # Criar arquivo Excel com múltiplas abas
                import io
                from pandas import ExcelWriter
                
                buffer = io.BytesIO()
                with ExcelWriter(buffer, engine='openpyxl') as writer:
                    desc_stats.to_excel(writer, sheet_name='Estatísticas_Descritivas', index=False)
                    test_results.to_excel(writer, sheet_name='Testes_Estatísticos', index=False)
                    benford_results.to_excel(writer, sheet_name='Lei_Benford', index=False)
                    
                    # Metadados e resumo
                    pd.DataFrame({
                        'Métrica': ['Total de Registros', 'Colunas Numéricas', 'Conforme Lei de Benford'],
                        'Valor': [
                            len(df_filtered), 
                            len(df_filtered.select_dtypes(include=['number']).columns),
                            'Sim' if benford_stats.get('benford_compliant', False) else 'Não'
                        ]
                    }).to_excel(writer, sheet_name='Resumo', index=False)
                
                file_data = buffer.getvalue()
                
                st.download_button(
                    label="Baixar resumo estatístico (Excel)",
                    data=file_data,
                    file_name="resumo_estatistico_eleicoes.xlsx",
                    mime="application/vnd.ms-excel"
                )

    # Adicionando a nova seção de Lei de Benford Avançada
    st.subheader("📊 Análise de Conformidade com a Lei de Benford")
    st.markdown("""
    A Lei de Benford é frequentemente utilizada em auditorias financeiras e análises eleitorais para identificar 
    possíveis manipulações nos dados. Ela estabelece que em muitos conjuntos naturais de dados, a distribuição do 
    primeiro dígito segue um padrão logarítmico específico.

    Esta análise avançada verifica se os dados eleitorais seguem a Lei de Benford e aplica testes estatísticos 
    para quantificar o nível de conformidade.
    """)

    # Opções de configuração para a análise
    benford_col1, benford_col2 = st.columns(2)
    with benford_col1:
        benford_column = st.selectbox(
            "Selecione a coluna para análise",
            ['QT_VOTOS', 'QT_ELEITORES'] if 'QT_ELEITORES' in df_filtered.columns else ['QT_VOTOS'],
            index=0
        )
        
    with benford_col2:
        benford_min = st.number_input(
            "Valor mínimo para análise",
            min_value=1,
            max_value=100,
            value=10,
            help="Filtra valores menores que este número para evitar distorções"
        )

    # Opção para agrupamento
    benford_agrupar = st.checkbox(
        "Agrupar dados antes da análise", 
        value=False,
        help="Agrupa os dados por município ou outro critério antes de aplicar a Lei de Benford"
    )

    benford_agrupamento = None
    if benford_agrupar:
        agrupamento_options = ['NM_MUNICIPIO', 'NR_ZONA', 'NR_LOCAL_VOTACAO']
        agrupamento_options = [opt for opt in agrupamento_options if opt in df_filtered.columns]
        if agrupamento_options:
            benford_agrupamento = st.selectbox("Agrupar por:", agrupamento_options)

    # Botão para realizar a análise
    if st.button("Executar Análise de Benford Avançada"):
        with st.spinner("Analisando conformidade com a Lei de Benford..."):
            benford_results = analyze_benford_law_advanced(
                df_filtered,
                column=benford_column,
                filter_min=benford_min,
                agrupamento=benford_agrupamento
            )
            
            if "error" in benford_results:
                st.error(benford_results["error"])
            else:
                # Criar gráfico comparativo
                fig = sp.make_subplots(
                    rows=2, cols=1,
                    subplot_titles=("Comparação com a Lei de Benford", "Desvio Percentual"),
                    vertical_spacing=0.15,
                    specs=[[{"type": "bar"}], [{"type": "bar"}]]
                )
                
                # Gráfico de barras comparando observado vs esperado
                fig.add_trace(
                    go.Bar(
                        x=benford_results['digits'],
                        y=benford_results['observed_freq'],
                        name="Observado",
                        marker_color='rgb(55, 83, 109)'
                    ),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=benford_results['digits'],
                        y=benford_results['expected_freq'],
                        name="Esperado (Benford)",
                        mode='lines+markers',
                        marker_color='red'
                    ),
                    row=1, col=1
                )
                
                # Gráfico do erro percentual
                fig.add_trace(
                    go.Bar(
                        x=benford_results['digits'],
                        y=benford_results['error_pct'],
                        name="Desvio (%)",
                        marker=dict(
                            color=benford_results['error_pct'],
                            colorscale='RdBu_r',
                            cmin=-100,
                            cmax=100
                        )
                    ),
                    row=2, col=1
                )
                
                # Adicionar linha zero no gráfico de desvio
                fig.add_shape(
                    type="line",
                    x0=0.5,
                    y0=0,
                    x1=9.5,
                    y1=0,
                    line=dict(color="black", width=1, dash="dot"),
                    row=2, col=1
                )
                
                # Ajustar layout e eixos
                fig.update_layout(
                    height=600,
                    width=800,
                    showlegend=True,
                    title_text="Análise da Lei de Benford",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
                )
                
                fig.update_xaxes(title_text="Primeiro Dígito", row=1, col=1)
                fig.update_yaxes(title_text="Frequência", row=1, col=1)
                fig.update_xaxes(title_text="Primeiro Dígito", row=2, col=1)
                fig.update_yaxes(title_text="Desvio Percentual", row=2, col=1)
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Exibir resultados dos testes estatísticos
                st.subheader("Resultados dos Testes Estatísticos")
                
                # Criar colunas para métricas
                metric_col1, metric_col2, metric_col3 = st.columns(3)
                
                with metric_col1:
                    conformidade = "Conforme" if benford_results['conformidade'] else "Não Conforme"
                    conformidade_delta = "✓" if benford_results['conformidade'] else "✗"
                    st.metric(
                        label="Conformidade com Benford",
                        value=conformidade,
                        delta=conformidade_delta
                    )
                
                with metric_col2:
                    chi2_formatted = f"{benford_results['chi2_stat']:.2f}"
                    st.metric(
                        label="Estatística χ²",
                        value=chi2_formatted,
                        delta=None,
                        help="Valores menores indicam melhor ajuste à distribuição esperada"
                    )
                
                with metric_col3:
                    p_value_formatted = f"{benford_results['p_value']:.4f}"
                    p_value_delta = "p > 0.05 ✓" if benford_results['p_value'] > 0.05 else "p < 0.05 ✗"
                    st.metric(
                        label="Valor-p",
                        value=p_value_formatted,
                        delta=p_value_delta,
                        help="Valor-p > 0.05 indica conformidade com a Lei de Benford"
                    )
                
                # Adicionar interpretação do MAD
                st.markdown(f"""
                **Desvio Absoluto Médio (MAD):** {benford_results['mad']:.4f}  
                **Interpretação:** {benford_results['mad_interpretation']}
                
                **Total de observações analisadas:** {benford_results['total_observations']:,}
                """)
                
                # Adicionar explicação contextual baseada nos resultados
                if benford_results['conformidade']:
                    st.success("""
                    ✅ **Interpretação:** Os dados analisados estão em conformidade com a Lei de Benford, 
                    sugerindo que os números seguem uma distribuição natural. Isso geralmente indica 
                    ausência de manipulação sistemática nos dados.
                    """)
                else:
                    st.warning("""
                    ⚠️ **Interpretação:** Os dados analisados apresentam desvios estatisticamente significativos 
                    da Lei de Benford. Isso não necessariamente indica fraude, mas sugere a necessidade de 
                    investigação adicional. Possíveis explicações incluem:
                    
                    - Manipulação intencional dos dados
                    - Características específicas do processo eleitoral
                    - Amostragem insuficiente ou enviesada
                    - Fatores contextuais específicos
                    
                    Recomenda-se realizar análises complementares para confirmar estes resultados.
                    """)

    # Adicionar nova seção na aba Machine Learning
    st.subheader("🔍 Agrupamento de Seções Eleitorais (Clustering)")
    st.markdown("""
    Esta análise agrupa as seções eleitorais com características semelhantes, permitindo:
    1. Identificar perfis típicos de comportamento das seções
    2. Detectar seções que se comportam de forma anômala em relação a seu grupo esperado
    3. Visualizar padrões geográficos de distribuição dos grupos

    Útil para identificar contextos locais e detectar anomalias contextuais que outros métodos podem não captar.
    """)

    # Configurações para o clustering
    clustering_col1, clustering_col2 = st.columns(2)

    with clustering_col1:
        # Obter todas as colunas numéricas disponíveis
        numeric_cols = [col for col in df_filtered.columns if df_filtered[col].dtype in ['int64', 'float64']]
        
        # Verificar quais colunas padrão estão disponíveis
        default_cols = []
        if 'QT_VOTOS' in numeric_cols:
            default_cols.append('QT_VOTOS')
        if 'QT_ELEITORES' in numeric_cols:
            default_cols.append('QT_ELEITORES')
        
        # Se não tiver nenhuma das colunas padrão, usar a primeira coluna numérica
        if not default_cols and numeric_cols:
            default_cols = [numeric_cols[0]]
        
        clustering_features = st.multiselect(
            "Selecione as características para agrupamento",
            options=numeric_cols,
            default=default_cols,
            help="Características numéricas que serão usadas para agrupar as seções"
        )

    with clustering_col2:
        n_clusters = st.slider(
            "Número de grupos (clusters)",
            min_value=2,
            max_value=10,
            value=5,
            help="Quantos grupos distintos serão identificados"
        )

    normalize = st.checkbox(
        "Normalizar dados",
        value=True,
        help="Recomendado quando as variáveis têm escalas diferentes"
    )

    # Executar clustering
    if st.button("Executar Análise de Agrupamento") and clustering_features:
        with st.spinner("Agrupando seções eleitorais com características semelhantes..."):
            df_clustered, model, metrics, centroids = perform_section_clustering(
                df_filtered,
                features=clustering_features,
                normalize=normalize,
                n_clusters=n_clusters
            )
            
            if df_clustered is None:
                st.error(f"Erro na análise de clustering: {metrics.get('error', 'Erro desconhecido')}")
            else:
                # Mostrar métricas do clustering
                st.subheader("Qualidade do Agrupamento")
                qual_col1, qual_col2 = st.columns(2)
                
                with qual_col1:
                    st.metric(
                        "Coeficiente de Silhueta",
                        f"{metrics['silhouette']:.3f}",
                        help="Mede a qualidade da separação dos clusters (quanto maior, melhor)"
                    )
                
                with qual_col2:
                    st.metric(
                        "Índice Calinski-Harabasz",
                        f"{metrics['calinski_harabasz']:.1f}",
                        help="Mede a razão entre dispersão entre e intra clusters (quanto maior, melhor)"
                    )
                
                # Visualizar características dos clusters
                st.subheader("Perfil dos Grupos Identificados")
                
                # Preparar e mostrar os centróides dos clusters
                fig_centroids = px.parallel_coordinates(
                    centroids,
                    color="CLUSTER",
                    labels={col: col.replace('_', ' ') for col in clustering_features},
                    color_continuous_scale=px.colors.diverging.Tealrose,
                    title="Características Médias de Cada Grupo"
                )
                st.plotly_chart(fig_centroids, use_container_width=True)
                
                # Distribuição do tamanho dos clusters
                cluster_sizes = pd.DataFrame({
                    'Cluster': list(metrics['cluster_sizes'].keys()),
                    'Quantidade de Seções': list(metrics['cluster_sizes'].values())
                })
                
                fig_sizes = px.bar(
                    cluster_sizes,
                    x='Cluster',
                    y='Quantidade de Seções',
                    title="Distribuição de Seções por Grupo",
                    color='Cluster',
                    color_continuous_scale=px.colors.sequential.Viridis
                )
                st.plotly_chart(fig_sizes, use_container_width=True)
                
                # Criar tabs para mais análises
                cluster_tabs = st.tabs([
                    "Visualização 3D", 
                    "Distribuição de Variáveis",
                    "Outliers", 
                    "Tabela Completa"
                ])
                
                # Aba 1: Visualização 3D (se tivermos 3 ou mais features)
                with cluster_tabs[0]:
                    if len(clustering_features) >= 3:
                        # Selecionar 3 features para visualização 3D
                        features_3d = clustering_features[:3]
                        
                        fig_3d = px.scatter_3d(
                            df_clustered,
                            x=features_3d[0],
                            y=features_3d[1],
                            z=features_3d[2],
                            color='CLUSTER',
                            opacity=0.7,
                            color_continuous_scale=px.colors.qualitative.G10,
                            title="Visualização 3D dos Grupos",
                            size_max=10,
                            width=800,
                            height=700
                        )
                        
                        # Melhorar a visualização 3D
                        fig_3d.update_layout(
                            scene=dict(
                                xaxis_title=features_3d[0],
                                yaxis_title=features_3d[1],
                                zaxis_title=features_3d[2]
                            )
                        )
                        
                        st.plotly_chart(fig_3d, use_container_width=True)
                    else:
                        # Criar um scatter plot 2D se tivermos menos de 3 features
                        if len(clustering_features) == 2:
                            fig_2d = px.scatter(
                                df_clustered,
                                x=clustering_features[0],
                                y=clustering_features[1],
                                color='CLUSTER',
                                opacity=0.7,
                                color_continuous_scale=px.colors.qualitative.G10,
                                title="Visualização 2D dos Grupos",
                                width=800,
                                height=600
                            )
                            st.plotly_chart(fig_2d, use_container_width=True)
                        else:
                            st.info("Selecione pelo menos 2 características para visualização espacial.")
                
                # Aba 2: Distribuição de variáveis por cluster
                with cluster_tabs[1]:
                    # Selecionar variável para visualização
                    var_to_plot = st.selectbox(
                        "Selecione a variável para visualizar a distribuição por grupo",
                        options=clustering_features
                    )
                    
                    # Criar boxplot para visualizar a distribuição
                    fig_box = px.box(
                        df_clustered,
                        x='CLUSTER',
                        y=var_to_plot,
                        color='CLUSTER',
                        title=f"Distribuição de {var_to_plot} por Grupo",
                        color_discrete_sequence=px.colors.qualitative.G10
                    )
                    st.plotly_chart(fig_box, use_container_width=True)
                    
                    # Adicionar histograma para melhor visualização
                    fig_hist = px.histogram(
                        df_clustered,
                        x=var_to_plot,
                        color='CLUSTER',
                        marginal="rug",
                        opacity=0.7,
                        barmode="overlay",
                        title=f"Histograma de {var_to_plot} por Grupo",
                        color_discrete_sequence=px.colors.qualitative.G10
                    )
                    st.plotly_chart(fig_hist, use_container_width=True)
                
                # Aba 3: Outliers dentro dos clusters
                with cluster_tabs[2]:
                    st.markdown("""
                    ### Outliers Dentro dos Grupos
                    
                    Estes são seções eleitorais que pertencem a um grupo, mas têm características 
                    muito diferentes das outras seções do mesmo grupo. 
                    
                    **Isso pode indicar anomalias contextuais** - seções que parecem normais 
                    numa análise global, mas são anômalas quando comparadas a seções similares.
                    """)
                    
                    # Mostrar estatísticas sobre outliers
                    n_outliers = df_clustered['OUTLIER_NO_CLUSTER'].sum()
                    pct_outliers = (n_outliers / len(df_clustered)) * 100
                    
                    st.metric(
                        "Seções Anômalas Dentro dos Grupos",
                        f"{n_outliers} ({pct_outliers:.1f}%)"
                    )
                    
                    # Filtrar e mostrar apenas os outliers
                    outliers_df = df_clustered[df_clustered['OUTLIER_NO_CLUSTER']]
                    
                    if len(outliers_df) > 0:
                        # Adicionar município aos dados, se disponível
                        display_cols = ['NM_MUNICIPIO', 'NR_ZONA', 'NR_SECAO', 'CLUSTER', 'DISTANCIA_CENTROIDE'] + clustering_features
                        display_cols = [col for col in display_cols if col in outliers_df.columns]
                        
                        st.dataframe(
                            outliers_df[display_cols].sort_values('DISTANCIA_CENTROIDE', ascending=False),
                            use_container_width=True
                        )
                        
                        # Download dos outliers
                        csv = outliers_df.to_csv(index=False)
                        st.download_button(
                            label="Baixar lista de seções anômalas como CSV",
                            data=csv,
                            file_name="secoes_anomalas_clustering.csv",
                            mime="text/csv"
                        )
                    else:
                        st.info("Nenhum outlier significativo encontrado dentro dos grupos.")
                
                # Aba 4: Tabela completa com resultados
                with cluster_tabs[3]:
                    st.subheader("Dados Completos com Atribuição de Grupos")
                    
                    # Adicionar município aos dados, se disponível
                    display_cols = ['NM_MUNICIPIO', 'NR_ZONA', 'NR_SECAO', 'CLUSTER', 'DISTANCIA_CENTROIDE', 'OUTLIER_NO_CLUSTER'] + clustering_features
                    display_cols = [col for col in display_cols if col in df_clustered.columns]
                    
                    st.dataframe(
                        df_clustered[display_cols],
                        use_container_width=True
                    )
                    
                    # Download dos dados completos
                    csv = df_clustered.to_csv(index=False)
                    st.download_button(
                        label="Baixar dados completos com grupos como CSV",
                        data=csv,
                        file_name="secoes_com_grupos.csv",
                        mime="text/csv"
                    )

if __name__ == '__main__':
    main() 
