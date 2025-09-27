# EigenBench

## Pipeline

``evaluations.py`` contains the framework for generating r_{ijk} samples on a given constitution, a dataset of scenarios, a list of models, and a list of personas.

``BT.py`` implements the vector Bradley-Terry-Davidson model, converting the evaluation scores r_{ijk} in {1,2} to scores in {0,1}, then training the judge lenses u_i, model dispositions v_j, and tie propensities lambda_i using CE loss.

- ``BT_criteria.py`` extends this code to judge comparison data that was collected comparison-wise (i.e. judge outputs one comparison per criterion of a constitution).

- ``BT_length.py`` extends this code to training a BTD model that is sensitive to the length of the models' responses

- ``BT_probs.py`` extends this code to training a BTD model that considers the probabilities of the judge's output.

``eigentrust.py`` runs EigenTrust on trained parameters u_i, v_j, lambda_i: forming the trust matrix then running fixed point iteration to find the principal left eigenvector.

``evaluations_MMLU.py`` and ``evaluations_GPQA.py`` contain the framework for running EigenBench evaluations on MMLU and GPQA datasets.

## Data
``utils.py`` details the functions used to made API calls to various labs' models.

``data_utils.py`` details the functions used to convert evaluations into pairwise comparison data.

``config.py`` contains all the constitutions and criteria used in our experiments.

``data`` contains all the scenario datasets.

``transcript`` is the main output folder of ``evaluation.py`` which contains all of the raw evaluation data and trained models on those experiments.

``transcript_MMLU`` and ``transcript_GPQA`` are outputs of ``evaluations_MMLU.py`` and ``evaluations_GPQA.py``.

``transcript_human`` contains data collected from human validation experiments.

## Notebooks
All ``.ipynb`` notebooks were used to run other experiments or visualizations: bootstraps, judge quality tests, embedding plots, elo score conversion, baseline surveys, etc.

``figures``, ``bootstrap``, and ``baselines`` contain some of the outputs of these notebooks.