# EigenBench

## Pipeline

``evaluations.py`` contains the framework for generating r_{ijk} samples on a given constitution, a dataset of scenarios, a list of models, and a list of personas.
- ``get_evaluations`` was the first attempt, which generated for every scenario and constitution a sample for every judge i and pair of evaluees j\<k.
- ``get_evaluation`` was found to be more effective on large datasets of scenarios, which instead randomly samples a scenario, judge, and pair of evaluees to perform an evaluation.
- ``get_model_evaluation`` modified the previous function to involve multiple different models (before the 'models' were simply the same model with different persona pre-prompts)
- ``get_model_evaluation_mn`` incorporated personas into the previous function, so that each i,j,k was sampled from a dataset of m models and n personas.

``BT.py`` implements the vector Bradley-Terry model, converting the evaluation scores r_{ijk} in {1,2} to scores in {0,1}, then training the judge u_i and model disposition v_j lenses using BCE loss.

``eigentrust.py`` runs EigenTrust on trained vectors u_i and v_j, forming the row-normalized trust matrix S_{ij} = exp(u_i dot v_j) / sum_k(exp(u_i dot v_k)) and then running iterations to find the principal left eigenvector.

## Other Tests
``evaluations_MMLU.py`` contains the framework for the test on the MMLU dataset. This is different from ``evaluations.py`` in the choice of models (which were cherry-picked by performance) and the evaluee and judge prompts, which required more context to answer the question.
