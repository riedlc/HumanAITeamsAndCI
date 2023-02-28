# Collective Intelligence In Human-AI Teams: A Bayesian Theory of Mind Approach

This is code for the following publication:
Westby, S. & Riedl, C. (2023). Collective Intelligence in Human-AI Teams: A Bayesian Theory of Mind Approach. In Proceedings of the 37th AAAI Conference on Artificial Intelligence (2023).


Workflow:
1. Download this directory
1. Change working directory to the downloaded `HumanAITeamsAndCI` directory
1. Create a Python 3.x environment and run `pip install -r requirements.txt`
1. Run `python -m scripts/results_table` to replicate Table 1  
1. Run `python -m scripts/figure_3` to replicate Figure 3  
        *Note:* There was a problem with the random seeds so results and figures are marginally different
1. Run `python -m scripts/grid_search` to redo the parameter grid search  
    - Each run requires approximately 50 GB
    - On an intel i7 with 8 GB of RAM, this takes approximately 40 minutes
    - Change `lower` and `upper` in `main()` to search a smaller parameter space
    - Run it twice. Once with  `self_actualize = True` and another with `self_actualize = False`
1. Now we move on the the R code - replicating Figure 4 and the log likelihoods in Table 1
1. In your preferred R editor, run `scripts/LikAnalysis_discovery.R`
    - Line 3 and 76 require you to input preferred directories
    - This file aggregates the grid search data
    - Repeat to analyze both grid searches from above
1. Run `rcode/LikAnalysis.R` to generate the logLiks from Table 1
    - Input your preferred directories where noted
1. Run `rcode/MessageToM.R` to generate the plots in Figure 4
    - Input your preferred directories where noted
  
