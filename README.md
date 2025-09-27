# EigenBench

## Pipeline

``evaluations.py`` contains the framework for generating r_{ijk} samples on a given constitution, a dataset of scenarios, a list of models, and a list of personas.

``BT.py`` implements the vector Bradley-Terry-Davidson model, converting the evaluation scores r_{ijk} in {1,2} to scores in {0,1}, then training the judge lenses u_i, model dispositions v_j, and tie propensities lambda_i using CE loss.

``eigentrust.py`` runs EigenTrust on trained parameters u_i, v_j, lambda_i: forming the trust matrix then running fixed point iteration to find the principal left eigenvector.

``evaluations_MMLU.py`` contains the framework for the test on the MMLU dataset. This is different from ``evaluations.py`` in the choice of models and the evaluee and judge prompts, which were constructed in separate contexts.

## Data
``util.py`` details the functions used to made API calls to various labs' models.

``config.py`` contains all the constitutions and criteria used in our experiments.

``data`` contains all the scenario datasets.

``transcript`` and ``transcript_MMLU`` are the main output folders of ``evaluation.py`` which contain all of the raw evaluation data and trained models on those experiments.

## Notebooks
All ``.ipynb`` notebooks were used to run other experiments or visualizations: bootstraps, judge quality tests, embedding plots, elo score conversion, baseline surveys, etc.

``figures``, ``bootstrap``, and ``baselines`` contain some of the outputs of these notebooks.