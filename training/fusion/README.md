# Fusion Model

This stage learns the final verdict from branch outputs and metadata features.

## Typical inputs

- image branch score
- face branch score
- video branch score
- heuristic detector scores
- platform or compression features

## Recommended first model

- logistic regression

Then upgrade later if needed:

- XGBoost
- small MLP

## Outputs

- `fusion.pkl`
- `feature_schema.json`
