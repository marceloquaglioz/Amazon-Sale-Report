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

## Pré Processamento
# Exclui coluna com erros e desnecerrária
amazon_sales.drop(columns=['index','Unnamed: 22'], inplace=True)

# Renomeia coluna com espaço no titulo
amazon_sales.rename(columns={'Sales Channel ': 'Sales Channel'}, inplace=True)

# Define o range de datas que será utilizado. Volumes inconsistentes fora desse range
amazon_sales = amazon_sales.query('Date >= "2022-04-04" & Date <= "2022-06-26"')

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

## Análise dos dados do valor das vendas
# Criar subplots
fig, axs = plt.subplots(1, 2, figsize=(14, 7))

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

# Análise das vendas por período

# Calcular as vendas totais por dia
df_diaria = amazon_sales.groupby(pd.Grouper(key='Date', freq='D'))['Amount'].sum().reset_index()
# Calcular as vendas totais por semana
df_semanal = amazon_sales.groupby(pd.Grouper(key='Date', freq='W'))['Amount'].sum().reset_index()
# Calcular as vendas totais por mês
# df_mensal = amazon_sales.groupby(pd.Grouper(key='Date', freq='ME'))['Amount'].sum().reset_index()

# Criar subplots
fig, axs = plt.subplots(1, 2, figsize=(14, 7))

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
# axs[2].plot(df_mensal['Date'], df_mensal['Amount'], marker='o')
# axs[2].set_title('Valor de Vendas Totais por Mês')
# axs[2].set_xlabel('Data')
# axs[2].grid(True)
# axs[2].tick_params(axis='x', rotation=45)
# axs[2].yaxis.set_major_formatter(FuncFormatter(millions))

# Ajustar layout
plt.tight_layout()
plt.show()

# Decomposição Sazonal
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
plt.figure(figsize=(14, 7))

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

# Calcular o Indice de Sazonalidade do valor das vendas 
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
plt.figure(figsize=(14, 7))
plt.plot(media_semanal['Date'], media_semanal['Índice de Sazonalidade'], marker='o', linestyle='-')
plt.xlabel('Semana')
plt.ylabel('Índice de Sazonalidade')
plt.title('Evolução do Índice de Sazonalidade Semanal')
plt.grid(True)
plt.tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.show()

## Análise das vendas por canal
# Calcular o volume financeiro de vendas por Fulfilment
sales_by_fulfilment = amazon_sales.groupby('Fulfilment')['Amount'].sum().reset_index()

fig, axes = plt.subplots(1, 3, figsize=(14, 7))

# Gráfico de Barras
axes[0].bar(sales_by_fulfilment['Fulfilment'], sales_by_fulfilment['Amount'], color='skyblue')
axes[0].set_xlabel('Fulfilment')
axes[0].set_ylabel('Volume Financeiro de Vendas')
# axes[0].set_title('Volume Financeiro de Vendas por Fulfilment')

# Adicionar os valores de cada barra
for bar in axes[0].patches:
    yval = bar.get_height()
    axes[0].text(bar.get_x() + bar.get_width()/2, yval, f'{yval/1e6:.2f}M', va='bottom', ha='center')

# Ajustar o eixo y para uma escala mais compreensível
axes[0].yaxis.set_major_formatter(FuncFormatter(millions))
axes[0].tick_params(axis='x', rotation=45)

# Boxplot
amazon_sales.boxplot(column='Amount', by='Fulfilment', grid=False, ax=axes[1])
axes[1].set_xlabel('Fulfilment')
axes[1].set_ylabel('Volume Financeiro de Vendas')
# axes[2].set_title('Distribuição do Volume Financeiro de Vendas por Fulfilment')
axes[1].tick_params(axis='x', rotation=45)

# Gráfico de Pizza
axes[2].pie(sales_by_fulfilment['Amount'], labels=sales_by_fulfilment['Fulfilment'], autopct='%1.1f%%', startangle=140, colors=plt.cm.Paired(range(len(sales_by_fulfilment))))
# axes[1].set_title('Distribuição do Volume Financeiro de Vendas por Fulfilment')

# Ajustar layout para evitar sobreposição
plt.suptitle('Distribuição do Volume Financeiro de Vendas por Fulfilment')
plt.tight_layout()
plt.show()

## Análise das vendas por categoria de produto
# Calcular o volume financeiro de vendas por Categoria
sales_by_category = amazon_sales.groupby('Category')['Amount'].sum().reset_index()

# Ordenar do maior para o menor
sales_by_category = sales_by_category.sort_values(by='Amount', ascending=False)

# Criar figura e eixos
fig, axes = plt.subplots(1, 3, figsize=(14 , 7))

# Gráfico de barras
axes[0].bar(sales_by_category['Category'], sales_by_category['Amount'], color='skyblue')
axes[0].set_xlabel('Categoria')
axes[0].set_ylabel('Volume Financeiro de Vendas')
# axes[0].set_title('Volume Financeiro de Vendas por Categoria')
axes[0].tick_params(axis='x', rotation=45)

# Adicionar os valores de cada barra em milhões
for bar in axes[0].containers[0]:
    yval = bar.get_height()
    axes[0].text(bar.get_x() + bar.get_width() / 2, yval, f'{yval / 1e6:.2f}M', va='bottom', ha='center')

# Ajustar o eixo y para uma escala mais compreensível
axes[0].yaxis.set_major_formatter(FuncFormatter(millions))

# Criar o boxplot
amazon_sales.boxplot(column='Amount', by='Category', grid=False, ax=axes[1])
axes[1].set_xlabel('Categoria')
axes[1].set_ylabel('Volume Financeiro de Vendas')
# axes[1].set_title('Distribuição do Volume Financeiro de Vendas por Categoria')
axes[1].tick_params(axis='x', rotation=45)
axes[1].set_suptitle('')

# Calcular o volume financeiro de vendas por Categoria
# sales_by_category = amazon_sales.groupby('Category')['Amount'].sum().reset_index()

# Calcular o percentual de cada categoria
total_amount = sales_by_category['Amount'].sum()
sales_by_category['Percentage'] = sales_by_category['Amount'] / total_amount * 100

# Agrupar categorias com percentual <= 5% em "Other"
other = sales_by_category[sales_by_category['Percentage'] <= 5].sum()
other['Category'] = 'Other'
other['Percentage'] = other['Amount'] / total_amount * 100

# Filtrar categorias com percentual > 5% e adicionar a categoria "Other"
sales_by_category = sales_by_category[sales_by_category['Percentage'] > 5]
sales_by_category = pd.concat([sales_by_category, other.to_frame().T], ignore_index=True)

# Ordenar as categorias do maior para o menor percentual
sales_by_category = sales_by_category.sort_values(by='Percentage', ascending=False)

# Criar o gráfico de pizza
axes[2].pie(sales_by_category['Amount'], labels=sales_by_category['Category'], autopct='%1.1f%%', startangle=140, colors=plt.cm.Paired(range(len(sales_by_category))))
# axes[2].set_title('Distribuição do Volume Financeiro de Vendas por Categoria')
axes[2].axis('equal')  # Assegura que o gráfico de pizza seja desenhado como um círculo.

plt.suptitle('Distribuição do Volume Financeiro de Vendas por Categoria')
plt.tight_layout()
plt.show()

## Análisendo os outliers que observamos nas anallises anteriores
# Calcular os quartis e o IQR (Interquartile Range)
Q1 = amazon_sales['Amount'].quantile(0.25)
Q3 = amazon_sales['Amount'].quantile(0.75)
IQR = Q3 - Q1

# Definir os limites inferior e superior para identificar os outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Identificar os outliers
outliers = amazon_sales[(amazon_sales['Amount'] < lower_bound) | (amazon_sales['Amount'] > upper_bound)]

# Criar uma nova coluna 'B2B_SimNao' com valores "Sim" ou "Não" baseados na coluna 'B2B'
outliers['B2B_SimNao'] = outliers['B2B'].apply(lambda x: 'Sim' if x else 'Não')

# Criar a nova coluna 'Promocao' com base na coluna 'promotion-ids'
outliers['Promocao'] = outliers['promotion-ids'].apply(lambda x: 'Sim' if pd.notnull(x) else 'Não')

print(f"Total de outliers encontrados: {len(outliers)}")
print(outliers[['Order ID', 'Date', 'Category', 'Fulfilment', 'B2B', 'SKU', 'Amount']])

# Estatísticas descritivas dos outliers
outliers_descriptive = outliers.describe()

print(outliers_descriptive)

# Configurar a figura e os subplots
fig, axs = plt.subplots(2, 2, figsize=(14, 7))

# Histograma dos outliers
axs[0, 0].hist(outliers['Amount'], bins=50, color='skyblue', edgecolor='black')
axs[0, 0].set_xlabel('Volume Financeiro (Amount)')
axs[0, 0].set_ylabel('Frequência')
# axs[0, 0].set_title('Distribuição dos Outliers de Volume Financeiro de Vendas')

# Scatter plot dos outliers por Categoria
axs[0, 1].scatter(outliers['Category'], outliers['Amount'], color='red', alpha=0.6)
axs[0, 1].set_xlabel('Categoria')
axs[0, 1].set_ylabel('Volume Financeiro (Amount)')
# axs[0, 1].set_title('Outliers de Volume Financeiro de Vendas por Categoria')
axs[0, 1].tick_params(axis='x', rotation=45)

# Scatter plot dos outliers por Fulfilment
axs[1, 0].scatter(outliers['Fulfilment'], outliers['Amount'], color='blue', alpha=0.6)
axs[1, 0].set_xlabel('Fulfilment')
axs[1, 0].set_ylabel('Volume Financeiro (Amount)')
# axs[1, 0].set_title('Outliers de Volume Financeiro de Vendas por Fulfilment')
axs[1, 0].tick_params(axis='x', rotation=45)

# Scatter plot dos outliers por B2B
axs[1, 1].scatter(outliers['B2B_SimNao'], outliers['Amount'], color='green', alpha=0.6)
axs[1, 1].set_xlabel('Cliente B2B')
axs[1, 1].set_ylabel('Volume Financeiro (Amount)')
# axs[1, 1].set_title('Outliers de Volume Financeiro de Vendas por B2B')
axs[1, 1].tick_params(axis='x', rotation=45)

# Ajustar o layout para evitar sobreposição
plt.suptitle('Análise de Outliers')
plt.tight_layout()
plt.show()

# Verificar concentração de outliers por data
outliers_date = outliers.groupby('Date').size().reset_index(name='Count')

# Analisar se os outliers estão associados a promoções específicas
outliers_promotion = outliers['Promocao'].value_counts()

# Verificar padrão específico no campo ship-service-level
outliers_ship_service = outliers['Status'].value_counts()

# Configurar a figura e os subplots
fig, axs = plt.subplots(1, 3, figsize=(14 , 7))

# Gráfico de linha para concentração de outliers por data
axs[0].plot(outliers_date['Date'], outliers_date['Count'], marker='o')
axs[0].set_xlabel('Data')
axs[0].set_ylabel('Número de Outliers')
axs[0].set_title('Concentração de Outliers por Data')
axs[0].tick_params(axis='x', rotation=45)

# Gráfico de barras para outliers associados a promoções específicas
axs[1].bar(outliers_promotion.index, outliers_promotion.values, color='green')
axs[1].set_xlabel('Promoção')
axs[1].set_ylabel('Número de Outliers')
axs[1].set_title('Outliers Associados a Promoções Específicas')
axs[1].tick_params(axis='x', rotation=90)

# Gráfico de barras para outliers por status do pedido
axs[2].bar(outliers_ship_service.index, outliers_ship_service.values, color='purple')
axs[2].set_xlabel('Status do Pedido')
axs[2].set_ylabel('Número de Outliers')
axs[2].set_title('Outliers por Status de Pedido')
axs[2].tick_params(axis='x', rotation=90)

# Ajustar o layout para evitar sobreposição
plt.suptitle('Análise de Outliers')
plt.tight_layout()
plt.show()
