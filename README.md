# WWTP Ammonium Prediction Model Comparison with Data Assimilation

## Introduction

Effective control and optimization of Wastewater Treatment Plants (WWTPs) are essential for meeting environmental regulations, but these are complex systems. Accurate prediction of critical variables like ammonium concentration $(NH_4^+)$ is fundamental for developing robust control strategies and ensuring efficient plant operation.

Modeling these dynamic processes is challenging due to system complexity, variable influents, and often limited or noisy sensor data. This work tackles these issues by systematically comparing white-box, grey-box, and black-box modeling approaches. We place particular emphasis on integrating **data assimilation techniques** to robustly handle data limitations, enabling simultaneous estimation of parameters and hidden states, and crucially, to quantify prediction **uncertainty** – a vital aspect often neglected.

Our study demonstrates through simulation that non-parametric black-box models combined with data assimilation achieve superior predictive accuracy and uncertainty quantification, even with sparse data. Real-world experiments confirm the approach's potential while highlighting the impact of input variability. We also found additional sensors may offer limited benefit if the target variable is already measured. Ultimately, this research provides insights into model selection for WWTP control under practical constraints and underscores the importance of data assimilation.

Further details, including the specific methodologies and full results, are presented in the article:

📝 **Data assimilation for prediction of ammonium in wastewater treatment plant: from
physical to data driven models**  
📚 *Authored by Victor BERTRET, Roman Le Goff Latimier and Valérie Monbet*  
🗖️ *Submitted to Water Research*  
🔗 [Link to paper (waiting)]

## Code Repository

The source code in this repository, `ASP_model_comparison`, was developed to implement the different model structures (white-box, grey-box, black-box), integrate the data assimilation techniques, and generate the comparative results presented in the aforementioned article.

### Organization

The project code is structured as follows:

* `data/`: Stores all data related to the experiments.
    * `dataset/`: Contains the raw datasets (simulated or real) used for training and evaluation.
    * `training/`: Holds the results from model training runs (e.g., estimated parameters, model states).
    * `rolling_evaluation/`: Contains results from the rolling window forecast evaluations.
* `plots/`: Directory where generated plots and figures are saved.
* `scripts/`: Contains executable Julia scripts to run the workflow.
    * `tasks/`: Main scripts for performing core experimental tasks. Use the `-h` flag for detailed options.
        * `generate_dataset.jl`: Creates simulated datasets (e.g., based on ASM1) or processes real data.
        * `train.jl`, `distributed_train.jl`: Executes model training/parameter estimation for different models and datasets. Saves results to `data/training/`. (Distributed version for parallel processing).
        * `rolling_test.jl`, `distributed_rolling_test.jl`: Performs rolling forecast evaluations using trained models from `data/training/`. Saves metrics to `data/rolling_evaluation/`. (Distributed version for parallel processing).
        * `rolling_results.jl`: Aggregates and compares results from `data/rolling_evaluation/` to generate summary figures or tables.
    * `supplementary_materials/`: Includes scripts for auxiliary analyses presented in the paper (e.g., `test_nn.jl`).
* `src/`: Contains the core Julia source code modules used by the scripts.
    * `dataset/`: Modules related to data loading, generation, and preprocessing.
    * `model/`: Implementation of the different predictive models (white-box, grey-box, black-box) and data assimilation algorithms.
    * `tasks/`: Underlying functions implementing the logic for training, evaluation, etc., called by the scripts in `scripts/tasks/`.
    * `utils/`: Helper functions, e.g., for processing results or plotting utilities.

### Installation

This code base is using the [Julia Language](https://julialang.org/) and
[DrWatson](https://juliadynamics.github.io/DrWatson.jl/stable/)
to make a reproducible scientific project named
> ASP_model_comparison

It is authored by Victor Bertret, Roman Le Goff Latimier and Valérie Monbet.

To (locally) reproduce this project, do the following:

0. Download this code base. Notice that raw data are typically not included in the
   git-history and may need to be downloaded independently.
1. Open a Julia console and do:
   ```
   julia> using Pkg
   julia> Pkg.add("DrWatson") # install globally, for using `quickactivate`
   julia> Pkg.activate("path/to/this/project")
   julia> Pkg.instantiate()
   ```

This will install all necessary packages for you to be able to run the scripts and
everything should work out of the box, including correctly finding local paths.

You may notice that most scripts start with the commands:
```julia
using DrWatson
@quickactivate "ASP_model_comparison"
```
which auto-activate the project and enable local path handling from DrWatson. For more details, see the [DrWatson repository](https://juliadynamics.github.io/DrWatson.jl/stable/).

### Usage Guide

After following the [Installation](#Installation) steps, you can run the different stages of the experimental workflow using the scripts located in the `scripts/tasks/` directory.

Most scripts accept command-line arguments (`kwargs`) to customize their behavior (e.g., select datasets, model parameters, etc.). You can typically view the available options by running the script with the `-h` or `--help` flag:

```bash
julia scripts/tasks/script_name.jl -h
```
(Replace script_name.jl with the actual script file name, e.g., train.jl)

Here is a typical workflow, based on the scripts described in the Organization section:

1. **Generate Datasets**: (If not already present or if customization is needed) Run the script to create simulated or process real datasets (here generate 50 times the S1 Dataset with different seeds).
   
    ```bash
    julia scripts/tasks/generate_dataset.jl S1 50
    ```
    Check the script's help `(-h)` for options related to dataset type (simplified ASM1, real) and parameters. Datasets are typically saved in `data/dataset`.

2. **Train Models**: Train the different models (white-box, grey-box, black-box with data assimilation) on the desired datasets. Specify model types, datasets, and training parameters via arguments (here train the linear black model on the S1 dataset and the first seed).

    ```bash
    julia scripts/tasks/train.jl -n S1 -s 1 linear_black
    # Or potentially for parallel execution (if configured):
    # julia scripts/tasks/distributed_train.jl [arguments...]
    ```
    Results (trained models/parameters) are saved in the `data/training` folder, likely organized by experiment parameters using DrWatson.

3. **Evaluate Models (Rolling Forecast)**: Perform rolling window predictions and calculate performance metrics using previously trained models (here test the linear black model on the S1 dataset and the first seed).

    ```bash
    julia scripts/tasks/rolling_test.jl -n S1 -s 1 linear_black
    # Or potentially for parallel execution (if configured):
    # julia scripts/tasks/distributed_rolling_test.jl [arguments...]
    ```

    This script retrieves trained models from `data/training` based on specified parameters and saves evaluation results in `data/rolling_evaluation`.

4. **Analyze and Compare Results**: Generate comparison plots and summaries based on the evaluation results. 

    ```bash
    julia scripts/tasks/rolling_results.jl
    ```

    This script processes all the data from `data/rolling_evaluation` and saves plots, likely in the plots directory.