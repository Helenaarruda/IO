# Equilibrium-matching estimates

The estimator solves the model equilibrium at each parameter guess and minimizes squared share discrepancies plus a weighted fit to observed prices of active products across all six datasets. Set `--price-weight 0` to use shares only.

| parameter          |   estimate |
|:-------------------|-----------:|
| beta_w             |     -0.095 |
| beta_g             |      0.195 |
| gamma_w            |      0.24  |
| gamma_g            |      0.295 |
| mu (scenario -0.5) |     -0.45  |
| mu (scenario 0)    |      0.05  |
| mu (scenario +0.5) |      0.45  |

## Fit by pricing regime

| pricing_regime   |   share_SSE |   share_RMSE |   active_price_SSE |   active_price_RMSE |   share_moments |   active_price_moments |
|:-----------------|------------:|-------------:|-------------------:|--------------------:|----------------:|-----------------------:|
| Bertrand         |         nan |          nan |                nan |                 nan |             150 |                      0 |
| Collusion        |         nan |          nan |                nan |                 nan |             150 |                      0 |

- Objective SSE (shares plus weighted active-price residuals): `15000`
- Share SSE: `nan`
- Share RMSE: `nan`
- Active-price SSE: `nan`
- Active-price RMSE: `nan`
- Active-price residual weight: `1.0`
- Optimizer success: `True` (`gtol` termination condition is satisfied.)
- Outer evaluations: `9`
- Elapsed seconds: `0.1`

## Parameter restrictions and model limitation

No equality restrictions are imposed on beta_w, beta_g, gamma_w, or gamma_g. The equilibrium solver applies its existing entry and collusion rules as written. In particular, its collusive common-price block is activated only when a2 and b1 have equal quality, so the objective can change discontinuously when a parameter guess breaks that tie. Results should therefore be interpreted with this feature of the existing solver in mind.

## Notes

- This is a nested equilibrium estimator; it is slower than demand fitting conditional on observed prices.
- The objective uses share residuals and standardized residuals for observed active prices; set `--price-weight 0` to estimate from shares only.
- No bootstrap or standard errors are computed.
- Iteration history and the best parameter vector seen so far are checkpointed during optimization.
