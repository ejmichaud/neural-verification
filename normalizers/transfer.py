# import libraries for copying directories recursives
import os
import yaml

# define directories to copy

dataset_path = "../tasks/"
original_raw_models_path = "../tasks/"
original_processed_models_path = "./rnn_tests/raw_models/"

task_names_with_models = [folder for folder in os.listdir(original_raw_models_path) if os.path.isdir(os.path.join(original_raw_models_path, folder))]
task_names_with_create_datasets = [folder for folder in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, folder))]
task_names_with_datasets = [folder for folder in os.listdir(dataset_path) if os.path.isfile(os.path.join(dataset_path, folder, "data.pt"))]
with open("../search.yaml", 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
task_names_in_yaml = list(config.keys())

task_names = set(task_names_in_yaml)
no_create_dataset = set(task_names) - set(task_names_with_create_datasets)
no_dataset = set(task_names) - set(task_names_with_datasets)
no_model = set(task_names_in_yaml) - set(task_names_with_models)
if no_create_dataset:
    raise ValueError
if no_dataset:
    raise ValueError
if no_model:
    raise ValueError


print(len(task_names))

#target_dir = "rnn_tests/raw_models_full"
#hammered_dir = "rnn_tests/raw_models"
target_dir = "rnn_tests/processed_models"
hammered_dir = "rnn_tests/hammered_models"

os.system(f"mkdir {target_dir}") 

os.system(f"mkdir {hammered_dir}")

for task in task_names:
#    if os.path.exists(target_dir + f"/{task}/model_perfect.pt"):
#        os.system(f"mkdir {hammered_dir}/{task}") 
#        os.system(f"cp {target_dir}/{task}/model_perfect.pt {hammered_dir}/{task}/model_perfect.pt") 
    if os.path.exists(target_dir + f"/{task}/model_perfect_whiten_0.1_jnf_0.7_toeplitz_debias_0.1_quantize_0.01.pt"):
        os.system(f"mkdir {hammered_dir}/{task}") 
        os.system(f"cp {target_dir}/{task}/model_perfect_whiten_0.1_jnf_0.7_toeplitz_debias_0.1_quantize_0.01.pt {hammered_dir}/{task}/model_hammered.pt") 
    else:
        print(f"{task} does not exist")
