# Função para formatar os eixos y em milhares (K)
def thousands(x, pos):
    return '%1.0fK' % (x * 1e-3)

# Função para formatar os eixos y em milhões (M)
def millions(x, pos):
    return '%1.1fM' % (x * 1e-6)
