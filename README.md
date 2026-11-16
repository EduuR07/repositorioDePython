# Trabalho A1 - Segmentação de clientes com GMM

Este repositório contém o trabalho desenvolvido para a disciplina de Aprendizagem de Máquina, utilizando o algoritmo **Gaussian Mixture Model (GMM)** para identificar diferentes perfis de compradores.

A ideia foi analisar os gastos de clientes de uma distribuidora e verificar se seria possível encontrar grupos com hábitos de consumo semelhantes, sem utilizar classificações prontas durante o treinamento.

## Sobre o projeto

Utilizamos a base de dados *Wholesale Customers*, disponibilizada pela UCI Machine Learning Repository.

A base possui 440 registros e informações sobre os gastos anuais dos clientes em seis categorias:

- Fresh (produtos frescos)
- Milk (leite e derivados)
- Grocery (mercearia)
- Frozen (congelados)
- Detergents_Paper (detergentes e papel)
- Delicassen (delicatessen)

Também existem informações sobre o canal de venda e a região de cada cliente, mas essas duas colunas foram utilizadas apenas para analisar os resultados depois do treinamento.

## Como o trabalho foi desenvolvido

O projeto foi feito em Python, utilizando principalmente as bibliotecas Pandas, NumPy, Matplotlib e Scikit-learn.

Primeiro, analisamos a base para verificar os dados disponíveis, possíveis valores ausentes e diferenças nos gastos dos clientes.

Depois, aplicamos uma transformação logarítmica (`log1p`) para reduzir a influência de valores muito altos e utilizamos o `StandardScaler` para padronizar os dados.

Separamos 80% dos registros para treinamento e 20% para teste, tomando cuidado para ajustar a padronização somente com os dados de treino.

Na parte de modelagem, testamos 24 configurações do GMM, variando a quantidade de componentes de 1 a 6 e utilizando quatro tipos de covariância: `full`, `tied`, `diag` e `spherical`.

Para comparar os modelos, utilizamos BIC, AIC e log-verossimilhança no conjunto de teste.

## Resultados

O menor BIC foi obtido com **3 componentes e covariância full**, configuração que escolhemos para a análise final.

Depois de ajustar o modelo novamente com os 440 clientes, utilizando 20 inicializações, encontramos três grupos:

- Grupo 0: 61 clientes, com gastos menores em várias categorias.
- Grupo 1: 222 clientes, com maior destaque para produtos frescos e congelados.
- Grupo 2: 157 clientes, com gastos mais elevados em leite, mercearia e produtos de limpeza.

Também analisamos as probabilidades de cada cliente pertencer aos grupos. Essa é uma das características do GMM, já que um cliente não precisa apresentar um comportamento totalmente definido para um único perfil.

Esses resultados ajudam a entender os padrões presentes na base, mas não significam que os grupos sejam classificações definitivas dos clientes.

## Como executar

O código está no arquivo `trabalhoDePython.ipynb` e pode ser executado pelo Google Colab ou por um ambiente Jupyter com as bibliotecas necessárias instaladas.

Para executar pelo Colab:

1. Abra o arquivo `trabalhoDePython.ipynb` no Google Colab.
2. Baixe a base [Wholesale Customers](https://archive.ics.uci.edu/dataset/292/wholesale+customers) no site da UCI.
3. Execute as células na ordem apresentada.
4. Quando o código solicitar, envie o arquivo `Wholesale_customers_data.csv`.
5. Acompanhe as tabelas, métricas e gráficos gerados durante a execução.

O notebook também gera arquivos CSV com a comparação dos modelos e a classificação dos clientes.

## Referências

Base de dados: [Wholesale Customers - UCI](https://archive.ics.uci.edu/dataset/292/wholesale+customers)

Documentação do algoritmo: [GaussianMixture - Scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html)
