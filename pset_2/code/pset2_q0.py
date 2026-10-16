
import jax
import jax.numpy as jnp
import pandas as pd
from pathlib import Path
from scipy.optimize import root
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import pandas as pd



# ==============================================================================
# Functions from Pietro's code
# ==============================================================================


# Este caminho é construído a partir da localização do script, portanto funciona
# independentemente da pasta em que o comando Python for executado.
#INPUT_DIR = Path(__file__).resolve().parents[1] / "data" / "input"
INPUT_DIR = Path.cwd() / "pset_2" / "data" / "input"


def carregar_datasets() -> dict[str, pd.DataFrame]:
    """Carrega os arquivos em um dicionário de DataFrames."""
    return {
        "choice_noodles": pd.read_csv(INPUT_DIR / "choice_noodles.csv"),
        "survey_noodles": pd.read_csv(INPUT_DIR / "survey_noodles.csv", sep="\t"),
        "college_year": pd.read_excel(INPUT_DIR / "college-year.xls"),
    }


datasets = carregar_datasets()

choice_noodles = datasets["choice_noodles"]
survey_noodles = datasets["survey_noodles"]
college_year = datasets["college_year"]


def montar_tabelas_cardapio(
    dados: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Resume a disponibilidade dos pratos por ano e marca."""
    opcoes_unicas = (
        dados[["year", "chain", "dish"]]
        .drop_duplicates()
        .sort_values(["year", "chain", "dish"])
    )

    # Uma linha para cada ano e marca, com os pratos disponíveis naquele ano.
    cardapio_por_ano = (
        opcoes_unicas.groupby(["year", "chain"], as_index=False)
        .agg(dishes_disponiveis=("dish", ", ".join))
        .sort_values(["year", "chain"])
    )

    # Primeiro e último ano observado para cada combinação marca--prato.
    entrada_dish = (
        opcoes_unicas.groupby(["chain", "dish"], as_index=False)
        .agg(primeiro_ano=("year", "min"), ultimo_ano=("year", "max"))
        .sort_values(["primeiro_ano", "chain", "dish"])
    )

    entrada_gluten_free = entrada_dish.loc[
        entrada_dish["dish"].str.endswith("_gf")
    ].reset_index(drop=True)

    return cardapio_por_ano, entrada_dish, entrada_gluten_free


cardapio_por_ano, entrada_dish, entrada_gluten_free = montar_tabelas_cardapio(
    choice_noodles
)

# No VS Code, rode esta célula e abra essas variáveis no Data Viewer:
# cardapio_por_ano, entrada_dish e entrada_gluten_free
