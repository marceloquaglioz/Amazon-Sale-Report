# importando bibliotecas necessárias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# carregando datasets do projeto
amazon_sales = pd.read_csv("dataset\\Amazon Sale Report.csv", 
                           low_memory=False,
                           parse_dates=['Date'], date_format='%m-%d-%y',
                           dtype={
                               'ship-postal-code': 'object'
                           })
# cloud_warehouse_comparison = pd.read_csv("dataset\\Cloud Warehouse Compersion Chart.csv")
# expense_iigf = pd.read_csv("dataset\\Expense IIGF.csv")
# international_sales = pd.read_csv("dataset\\International sale Report.csv")
# may_2022_sales = pd.read_csv("dataset\\May-2022.csv")
# march_2021_sales = pd.read_csv("dataset\\P  L March 2021.csv")
# sale_report = pd.read_csv("dataset\\Sale Report.csv")

# Obtem informações da estrutura do dataframe
amazon_sales.info()

# Exclui coluna com erros e desnecerrárias
amazon_sales.drop(columns=['index','Unnamed: 22'], inplace=True)

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

amazon_sales[['Qty','Amount']].describe()
