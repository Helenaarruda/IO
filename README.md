# IO
IO 1 class files rep

Have the project folder as your working directory, and be sure to run Python from `IO/pset_1/.venv/bin/python`.

From that folder, with the `uv` package installed, run
```
uv sync
uv run python code/run_all.py
```
in the terminal.

The master script executes the following scripts in order:

    - `q1_p2.py`: generates simulated datasets and estimates the labor supply and labor demand models.
    - `solve_equilibria.py`: solves the equilibrium models.
    - `question_iii.py`: runs the computations for Question III.
    - `plot_market_shares.py`: generates market-share plots for both cases with (\mu=0).
    - `estimation.py`: runs the estimation procedure.
