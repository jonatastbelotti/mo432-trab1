import os
import csv
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


# DEFINIÇÕES
COLUNAS_NUMERICAS = ["Year", "Present_Price", "Kms_Driven", "Owner"]
COLUNAS_CATEGORICAS = ["Car_Name", "Fuel_Type", "Seller_Type", "Transmission"]
COLUNAS_ENTRADAS = COLUNAS_CATEGORICAS + COLUNAS_NUMERICAS
COLUNA_SAIDA = "Selling_Price"


# Função que carrega os dados do arquivo CVS
def carregar_dados(arquivo=None):
    '''
    Leia o arquivo car-data.csv. O arquivo é um csv com colunas que são strings (representando valores categóricos).
    Voce provavelmente precisa usar algo como o pandas para ler o arquivo como um todo.
    O atributo de saída é Selling_Price
    '''
    entradas = list()
    saidas = list()

    if not arquivo:
        arquivo = os.path.dirname(os.path.realpath(__file__)) + "/car-data.csv"

    with open(arquivo) as csv_file:
        csv_reader = csv.DictReader(csv_file, delimiter=',')
        for linha in csv_reader:
            entradas.append([float(linha[att]) if att in COLUNAS_NUMERICAS else linha[att] for att in COLUNAS_ENTRADAS])
            saidas.append(float(linha[COLUNA_SAIDA]))
        
        csv_file.close()
    
    print("Foram lidas %d linhas com %d colunas" % (len(entradas), len(COLUNAS_ENTRADAS) + 1))

    return entradas, saidas


# Converta os atributos categóricos para numéricos
def converter_dados(entradas):
    '''
    Usando o one-hot-enconder, converta todos os atributos categóricos para numéricos.
    '''
    resp = list([] for i in entradas)
    categoricas = list([] for i in entradas)

    # Percorre todas as colunas
    for i_coluna, coluna in enumerate(COLUNAS_ENTRADAS):
        # Se for uma coluna numérica apenas a adiciona
        if coluna in COLUNAS_NUMERICAS:
            for i_linha, linha in enumerate(entradas):
                resp[i_linha].append(linha[i_coluna])
        
        # Se for uma coluna categórica adiciona nas categóricas
        elif coluna in COLUNAS_CATEGORICAS:
            for i_linha, linha in enumerate(entradas):
                categoricas[i_linha].append(linha[i_coluna])
    
    # Aplica OneHotEncoder
    # https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(categoricas)
    for i_linha, linha in enumerate(enc.transform(categoricas).toarray()):
        resp[i_linha].extend(linha)

    return resp


# Faca o centering and standard scaling para todos os atributos de entrada
def centring_scaling(dados):
    '''
    O StandardScaler já aplica o Centering e o Scaling
    https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
    '''
    # numericos = dados[:, :len(COLUNAS_NUMERICAS)]
    # categoricos = dados[:, len(COLUNAS_NUMERICAS):]
    # scaler = StandardScaler()
    # scaler.fit(numericos)
    # numericos = scaler.transform(numericos)
    # return np.concatenate((numericos, categoricos), axis=1)

    scaler = StandardScaler()
    scaler.fit(dados)
    return scaler.transform(dados)



# Função que plota o gráfico com os valores acumulados de PCA
def plotar_grafico_pca(valores):
    valores_x = [i + 1 for i in range(len(valores))]
    plt.bar(valores_x, valores)
    plt.yticks([y for y in range(0, 101, 10)])

    plt.title("Componentes príncipais segundo PCA", fontweight='regular', color = 'black', fontsize='13', horizontalalignment='center')
    plt.xlabel('Número de componentes', fontweight='regular', color = 'black', fontsize='13', horizontalalignment='center')
    plt.ylabel('Variância acumulada', fontweight='regular', color = 'black', fontsize='13', horizontalalignment='center')
    plt.rcParams["figure.figsize"] = (16,9)

    plt.savefig("./grafico-pca.png", dpi=300)


# reduza a dimensionalidade dos atributos de entrada usando PCA.
def reduzir_dimensionalidade(dados):
    '''
    Quantas dimensões restarão se mantivermos 90% da variância dos dados?
    Use o scree plot para determinar quantas dimensões devem ser mantidas.
    Converta os dados usando o PCA com 90% das variância.
    '''
    # Cálcula PCA para cada coluna
    pca = PCA(n_components=dados.shape[1])
    pca.fit(dados)
    valores = list()
    val = 0
    
    # Calcula valor acumulado
    for v in pca.explained_variance_ratio_:
        val += v
        valores.append(val * 100)

    # Plotando gráfico com a variância acumulada
    plotar_grafico_pca(valores)

    # Reduzindo dimensionalidade para representar 90% das amostras
    pca = PCA(0.9)
    pca.fit(dados)
    resp = pca.transform(dados)

    print("A dimensionalidade foi reduzida de %d para %d atributos" % (dados.shape[1], pca.n_components_))

    return resp




# CÓDIGO PRINCIPAL
if __name__ == "__main__":
    # Carregando arquivo CSV com os dados
    entradas, saidas = carregar_dados()

    # Converte os dados categóricos em numéricos
    entradas = converter_dados(entradas)

    # Convertendo dados para arrays Numpy
    entradas = np.array(entradas)
    saidas = np.array(saidas)

    # Centering and scaling
    entradas = centring_scaling(entradas)

    # reduza a dimensionalidade dos atributos de entrada usando PCA.
    entradas = reduzir_dimensionalidade(entradas)


    aaaaaa = 1