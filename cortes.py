"""
Parte 1 - Gráfico de histogramas para análise de pontos de corte.

Uso como módulo:
    from cortes import escolher_cortes, avaliar_faixas, plotar_histogramas
Uso por linha de comando (CSV com colunas y_true e proba):
    python cortes.py --csv teste.csv --largura 10 --t1 0.2 --t2 0.8 --saida grafico.png
    (opcional: --rotulo-positivo "fraude", --col-y, --col-proba)

Convenções:
  * proba = probabilidade estimada da CLASSE POSITIVA para TODAS as instâncias.
  * y_true = 1 se o rótulo real é a classe positiva, 0 caso contrário.
  * Faixas: negativa automática (p < t1), manual (t1 <= p < t2),
            positiva automática (p >= t2).
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _validar(y, p):
    y = np.asarray(y).astype(int)
    p = np.asarray(p).astype(float)
    if y.shape != p.shape or y.ndim != 1:
        raise ValueError("y_true e proba devem ser vetores de mesmo tamanho.")
    if not set(np.unique(y)).issubset({0, 1}):
        raise ValueError("y_true deve conter apenas 0 e 1 (1 = classe positiva).")
    if np.isnan(p).any() or p.min() < 0 or p.max() > 1:
        raise ValueError("proba deve estar em [0, 1] e sem NaN.")
    return y, p


def _bordas(largura_pp):
    n = 100 / largura_pp
    if abs(n - round(n)) > 1e-9:
        raise ValueError("A largura (em pontos percentuais) deve dividir 100.")
    return np.linspace(0, 1, int(round(n)) + 1)


def histograma_percentual(y, p, largura_pp=10):
    """Percentual de instâncias por intervalo, sempre sobre o TOTAL N.
    Cada instância cai em exatamente um intervalo [a, b); o último inclui 1,0."""
    y, p = _validar(y, p)
    bordas = _bordas(largura_pp)
    idx = np.clip(np.digitize(p, bordas[1:-1], right=False), 0, len(bordas) - 2)
    n = len(p)
    todos = np.bincount(idx, minlength=len(bordas) - 1) / n * 100
    pos = np.bincount(idx[y == 1], minlength=len(bordas) - 1) / n * 100
    assert abs(todos.sum() - 100) < 1e-9
    return bordas, todos, pos


def avaliar_faixas(y, p, t1, t2):
    """Tabela por faixa + métricas de erro com denominadores explícitos."""
    y, p = _validar(y, p)
    if not t1 < t2:
        raise ValueError("É necessário t1 < t2.")
    n, n_pos, n_neg = len(y), int(y.sum()), int((1 - y).sum())
    faixas = {
        "Negativa automática (p < t1)": p < t1,
        "Análise manual (t1 <= p < t2)": (p >= t1) & (p < t2),
        "Positiva automática (p >= t2)": p >= t2,
    }
    linhas = []
    for nome, m in faixas.items():
        q = int(m.sum())
        qp, qn = int(y[m].sum()), int((1 - y[m]).sum())
        linhas.append({
            "Faixa": nome, "Instâncias": q, "% da população (de N)": 100 * q / n,
            "Positivos": qp, "Negativos": qn,
            "% positivos na faixa (de instâncias da faixa)": 100 * qp / q if q else np.nan,
            "% negativos na faixa (de instâncias da faixa)": 100 * qn / q if q else np.nan,
        })
    tab = pd.DataFrame(linhas)
    fn = int(y[p < t1].sum())
    fp = int((1 - y[p >= t2]).sum())
    erros = {
        "N total": n, "Positivos reais": n_pos, "Negativos reais": n_neg,
        "Positivos classificados como negativos (FN automáticos)": fn,
        "  % sobre todos os positivos reais": 100 * fn / n_pos if n_pos else np.nan,
        "  % sobre os classificados automaticamente como negativos":
            100 * fn / int((p < t1).sum()) if (p < t1).any() else np.nan,
        "Negativos classificados como positivos (FP automáticos)": fp,
        "  % sobre todos os negativos reais": 100 * fp / n_neg if n_neg else np.nan,
        "  % sobre os classificados automaticamente como positivos":
            100 * fp / int((p >= t2).sum()) if (p >= t2).any() else np.nan,
        "Cobertura automática (% de N)": 100 * (1 - faixas["Análise manual (t1 <= p < t2)"].mean()),
    }
    return tab, erros


def escolher_cortes(y_val, p_val, max_fn_auto=0.02, max_fp_auto=0.02, passo=0.01):
    """Escolhe t1 e t2 NA VALIDAÇÃO.
    t1: entre as instâncias com p < t1, a fração de positivos reais
        (FN entre as negativas automáticas) deve ser <= max_fn_auto.
    t2: entre as instâncias com p >= t2, a fração de negativos reais
        (FP entre as positivas automáticas) deve ser <= max_fp_auto.
    Entre todos os pares (t1 < t2) que respeitam as tolerâncias, escolhe o que
    minimiza a quantidade de instâncias enviadas à análise manual (maximiza a cobertura
    automática). Levanta ValueError se as tolerâncias forem inviáveis."""
    y, p = _validar(y_val, p_val)
    grade = np.round(np.arange(passo, 1.0, passo), 10)
    # candidatos a t1: toda a região abaixo de t (e de qualquer t' < t) respeita a tolerância
    A, ok = [], True
    for t in grade:
        m = p < t
        ok = ok and (not m.any() or y[m].mean() <= max_fn_auto)
        if ok:
            A.append(t)
    # candidatos a t2: toda a região acima de t (e de qualquer t' > t) respeita a tolerância
    B, ok = [], True
    for t in grade[::-1]:
        m = p >= t
        ok = ok and (not m.any() or (1 - y[m]).mean() <= max_fp_auto)
        if ok:
            B.append(t)
    melhor = None  # minimiza o volume da análise manual; desempate: faixa manual mais estreita
    for t1 in A:
        for t2 in B:
            if t1 < t2:
                n_man = int(((p >= t1) & (p < t2)).sum())
                chave = (n_man, t2 - t1)
                if melhor is None or chave < melhor[0]:
                    melhor = (chave, t1, t2)
    if melhor is None:
        raise ValueError("Tolerâncias inviáveis: não há t1 < t2 que as respeitem. Relaxe-as.")
    return float(melhor[1]), float(melhor[2])


def plotar_histogramas(y, p, largura_pp=10, t1=None, t2=None, arquivo=None,
                       rotulo_positivo="classe positiva", titulo=None):
    bordas, todos, pos = histograma_percentual(y, p, largura_pp)
    x = bordas[:-1] * 100
    w = largura_pp
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(x, todos, width=w, align="edge", color="blue", alpha=0.6,
           edgecolor="navy", label="Todas as instâncias")
    ax.bar(x, pos, width=w, align="edge", color="red", alpha=0.9,
           edgecolor="darkred", label=f"Rótulo real positivo ({rotulo_positivo})")
    for i in range(len(x)):
        if todos[i] > 0:
            ax.text(x[i] + w / 2, todos[i] + 0.3, f"{todos[i]:.1f}", ha="center", fontsize=8, color="navy")
    ax.set_ylim(0, max(todos.max() * 1.35, 1))
    if t1 is not None and t2 is not None:
        for t in (t1, t2):
            ax.axvline(t * 100, color="k", ls="--", lw=1.2)
        tab, _ = avaliar_faixas(y, p, t1, t2)
        topo = ax.get_ylim()[1]
        centros = [(0 + t1 * 100) / 2, (t1 + t2) * 50, (t2 * 100 + 100) / 2]
        nomes = ["Negativa\nautomática", "Análise\nmanual", "Positiva\nautomática"]
        for c, nm, (_, r) in zip(centros, nomes, tab.iterrows()):
            ax.text(c, topo * 0.97, f"{nm}\n{r['% da população (de N)']:.1f}% da pop.",
                    ha="center", va="top", fontsize=9, bbox=dict(fc="white", ec="gray", alpha=0.8))
    ax.set_xlim(0, 100)
    ax.set_xticks(np.arange(0, 101, max(largura_pp, 10)))
    ax.set_xlabel(f"Probabilidade estimada da classe positiva ({rotulo_positivo}) (%)")
    ax.set_ylabel("% de instâncias (sobre o total do conjunto)")
    ax.set_title(titulo or "Distribuição das probabilidades previstas (conjunto de teste)")
    ax.legend(loc="center right")
    fig.tight_layout()
    if arquivo:
        fig.savefig(arquivo, dpi=150)
    return fig


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("--csv", required=True)
    a.add_argument("--col-y", default="y_true")
    a.add_argument("--col-proba", default="proba")
    a.add_argument("--largura", type=float, default=10, help="largura em pontos percentuais")
    a.add_argument("--t1", type=float, required=True)
    a.add_argument("--t2", type=float, required=True)
    a.add_argument("--rotulo-positivo", default="classe positiva")
    a.add_argument("--saida", default="grafico_cortes.png")
    g = a.parse_args()
    df = pd.read_csv(g.csv)
    plotar_histogramas(df[g.col_y], df[g.col_proba], g.largura, g.t1, g.t2, g.saida, g.rotulo_positivo)
    tab, err = avaliar_faixas(df[g.col_y], df[g.col_proba], g.t1, g.t2)
    pd.set_option("display.width", 250, "display.max_columns", 20)
    print(tab.round(2).to_string(index=False))
    for k, v in err.items():
        print(f"{k}: {v:.2f}" if isinstance(v, float) else f"{k}: {v}")


if __name__ == "__main__":
    main()
