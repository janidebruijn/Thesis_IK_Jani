# Thesis_IK_Jani
Individual files of J.A.I. de Bruijn

**Studentnumber**: S5234328

## Repository Structure and File Descriptions

This repository contains all scripts used and all model outputs made conducting this research.

- `model_outputs`
  Directly contains the output of the large classification task, aswell as all model outputs on the test set (in nested folders).

- `scripts`
  Contains main classification scripts for all models, aswell as the general functions scripts containing functions used in the scripts for each individual model.

- `setup_and_run.sh`
  Bash script to create a job, set the environment and run the python scripts.

- `requirements.txt`
  Text file containing the required installs for this research.

- `gold_standard_test.csv` and `binary_gold_standard_test.csv`
  Both multi-class and binary gold standard test sets 

- `llama_token.txt`
  Text file that should contain your personal secret llama access token. (For legal reasons my personal access token cannot be found in there.)

- `merged_output.csv`
  CSV file containing the final classification output merged with corresponding delta awards.

- `csv_to_binary.py`
  Script used to convert the gold standard test set to a binary gold standard test set.

- `correlations.py`
  Script used to calculate the chi-square statistics between elements and delta (script originally created by M.R. Kooning, altered it to fit my research with her permission).

- `merge_files.py`
  Script used to merge the final classification output with corresponding delta awards.

- `metrics.py`
  Script used to create classification reports on the models outputs.

## Google Drive

All other shared files can be found on Google Drive. The file called `threads1000_format_preprocessed.csv` was used for labeling. The map called 'group1' contains all the files our group used together.

https://drive.google.com/drive/folders/1PclYOGt4jK8dUiOy74PvkWd5HfnA3uX6?usp=drive_link

## How to Conduct the Final Classification Task

1. Open a Hábrók terminal, create a folder and upload all files from this repository. (The `model_outputs` folder is not required.) Also upload `threads1000_format_preprocessed.csv` found in the Google Drive.
2. Create a virtual python environment: `python -m venv $HOME/venvs/thesis_env`
3. In this virtual environment, install requirements: `pip install -r requirements.txt`
4. Leave the virtual environment.
5. Run `sbatch setup_and_run.sh` to create and run the job and wait for it to complete (took over 15 hours).
6. Use `merge_files.py` to merge the output file with corresponding delta awards.
7. Use `correlations.py` on `merged_output.csv` to calculate the chi-square statistics between elements and delta.
