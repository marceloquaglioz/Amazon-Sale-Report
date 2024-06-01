# importando bibliotecas necessárias
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
# import seaborn as sns

from matplotlib.ticker import FuncFormatter
from src.utils import thousands, millions

# carregando datasets do projeto
amazon_sales = pd.read_csv("dataset\\Amazon Sale Report.csv", 
                           low_memory=False,
                           parse_dates=['Date'], date_format='%m-%d-%y',
                           dtype={
                               'ship-postal-code': 'object'
                           })

# Obtem informações da estrutura do dataframe
amazon_sales.info()

# Exclui coluna com erros e desnecerrárias
amazon_sales.drop(columns=['index','Unnamed: 22'], inplace=True)

# Define o range de datas que será utilizado. Volumes inconsistentes fora desse range
amazon_sales = amazon_sales.query('Date >= "2022-04-01" & Date < "2022-06-28"')

# Alterar os valores nulos em "Courier Status" de acordo com a condição especificada
mask = amazon_sales['Courier Status'].isnull()  # Encontra linhas com valores nulos em "Courier Status"
cancelled_mask = amazon_sales['Status'] == 'Cancelled'  # Encontra linhas onde "Status" é "Cancelled"
# Atribui "Cancelled" onde a máscara de status é True e a máscara de valores nulos é True
amazon_sales.loc[mask & cancelled_mask, 'Courier Status'] = 'Cancelled'
# Preenche os valores nulos restantes em "Courier Status" com "Other"
amazon_sales['Courier Status'] = amazon_sales['Courier Status'].fillna('Other')

# Altera valores em Currency
amazon_sales['currency'] = amazon_sales['currency'].fillna('-')

# Altera valores em Amount
amazon_sales['Amount'] = amazon_sales['Amount'].fillna(0)

# Altera valores das colunas ship-
amazon_sales = amazon_sales.fillna({
    'ship-city': '-',
    'ship-state': '-',
    'ship-postal-code': '-',
    'ship-country': '0'
})

# Remove valor .0 no campo ship-postal-code
amazon_sales['ship-postal-code'] = amazon_sales['ship-postal-code'].astype(str).str.rstrip('.0')

# Altera valores em fulfilled-by
amazon_sales['fulfilled-by'] = amazon_sales['fulfilled-by'].fillna('India Post')

# Estatisticas descritivas dos campos numéricos
amazon_sales[['Qty','Amount']].describe()

# Analise dos dados do valor das vendas
# Criar subplots
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Histograma
axs[0].hist(amazon_sales['Amount'], bins=30, color='blue', edgecolor='black')
axs[0].set_title('Histograma de Valor de Vendas')
axs[0].set_xlabel('Amount')
axs[0].set_ylabel('Frequência')

# Boxplot
axs[1].boxplot(amazon_sales['Amount'], vert=False)
axs[1].set_title('Boxplot de Valor de Vendas')
axs[1].set_xlabel('Amount')

# Ajustar layout
plt.tight_layout()
plt.show()

# Analise das vendas por período

# Calcular as vendas totais por dia
df_diaria = amazon_sales.groupby(pd.Grouper(key='Date', freq='D'))['Amount'].sum().reset_index()
# Calcular as vendas totais por semana
df_semanal = amazon_sales.groupby(pd.Grouper(key='Date', freq='W'))['Amount'].sum().reset_index()
# Calcular as vendas totais por mês
df_mensal = amazon_sales.groupby(pd.Grouper(key='Date', freq='ME'))['Amount'].sum().reset_index()

# Criar subplots
fig, axs = plt.subplots(1, 3, figsize=(18, 6))

# Gráfico de valor de vendas totais por dia
axs[0].plot(df_diaria['Date'], df_diaria['Amount'], marker='o')
axs[0].set_title('Valor de Vendas Totais por Dia')
axs[0].set_xlabel('Data')
axs[0].set_ylabel('Vendas Totais')
axs[0].grid(True)
axs[0].tick_params(axis='x', rotation=45)
axs[0].yaxis.set_major_formatter(FuncFormatter(thousands))

# Gráfico de vendas totais por semana
axs[1].plot(df_semanal['Date'], df_semanal['Amount'], marker='o')
axs[1].set_title('Valor de Vendas Totais por Semana')
axs[1].set_xlabel('Data')
axs[1].grid(True)
axs[1].tick_params(axis='x', rotation=45)
axs[1].yaxis.set_major_formatter(FuncFormatter(thousands))

# Gráfico de vendas totais por mês
axs[2].plot(df_mensal['Date'], df_mensal['Amount'], marker='o')
axs[2].set_title('Valor de Vendas Totais por Mês')
axs[2].set_xlabel('Data')
axs[2].grid(True)
axs[2].tick_params(axis='x', rotation=45)
axs[2].yaxis.set_major_formatter(FuncFormatter(millions))

# Ajustar layout
plt.tight_layout()
plt.show()

# Decomposição Sasonal
# Usando o Valor de Vendas Diario. O dataset possui apenas 3 meses
serie_temporal = df_diaria.set_index('Date')['Amount']

# Verificar a frequência dos dados
print(df_diaria['Date'].diff().mode())  # Verifica a moda da diferença de datas

# Decomposição sazonal clássica com período de 7 dias (semanal)
decomposicao = seasonal_decompose(serie_temporal, model='additive', period=7)

# Plotar os componentes da decomposição
# decomposicao.plot()
# plt.show()

# Obter os componentes da decomposição
componente_tendencia = decomposicao.trend
componente_sazonal = decomposicao.seasonal
componente_residuo = decomposicao.resid

# Mostrar componentes individualmente
plt.figure(figsize=(12, 8))

plt.subplot(411)
plt.plot(serie_temporal, label='Original')
plt.legend(loc='upper left')

plt.subplot(412)
plt.plot(componente_tendencia, label='Tendência')
plt.legend(loc='upper left')

plt.subplot(413)
plt.plot(componente_sazonal, label='Sazonalidade')
plt.legend(loc='upper left')

plt.subplot(414)
plt.plot(componente_residuo, label='Resíduo')
plt.legend(loc='upper left')

plt.tight_layout()
plt.show()

# Calcular o Indice de Sazonalidade
# Calcular a média geral das vendas
media_semanal = df_diaria.groupby(pd.Grouper(key='Date', freq='W'))['Amount'].mean().reset_index()
media_geral = media_semanal['Amount'].mean()

# Calcular o índice de sazonalidade semana a semana
indices_sazonalidade = []
for _, row in media_semanal.iterrows():
    vendas_semanais = row['Amount']
    amplitude_media_sazonalidade = abs(vendas_semanais - media_geral)
    indice_sazonalidade = amplitude_media_sazonalidade / media_geral
    indices_sazonalidade.append(indice_sazonalidade)

# Adicionar os índices de sazonalidade calculados ao DataFrame semanal
media_semanal['Índice de Sazonalidade'] = indices_sazonalidade

# Plotar o índice de sazonalidade ao longo das semanas
plt.figure(figsize=(10, 6))
plt.plot(media_semanal['Date'], media_semanal['Índice de Sazonalidade'], marker='o', linestyle='-')
plt.xlabel('Semana')
plt.ylabel('Índice de Sazonalidade')
plt.title('Evolução do Índice de Sazonalidade Semanal')
plt.grid(True)
plt.tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.show()

# Analise das vendas por canal
