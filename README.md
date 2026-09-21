# Trabalho A2 - Instruções

    pip install numpy pandas scikit-learn matplotlib shap
    # um comando por domínio (3 bases, 3 CSVs):
    python pipeline.py --csv spam.csv  --alvo label --positivo spam --nome spam --texto texto
    python pipeline.py --csv churn.csv --alvo Churn --positivo Yes  --nome churn
    python pipeline.py --csv cancer.csv --alvo target --positivo maligno --nome cancer

    # Parte 1 isolada, em qualquer modelo (CSV com y_true e proba da classe positiva):
    python cortes.py --csv resultados/spam/teste_probabilidades.csv --t1 0.2 --t2 0.8 --largura 10 --rotulo-positivo spam

Ajuste `--max-fn-auto` / `--max-fp-auto` conforme o custo dos erros do domínio.
Cada execução grava tabelas, curvas ROC/PR, matrizes de confusão, SHAP e o gráfico de cortes em resultados/<nome>/.
