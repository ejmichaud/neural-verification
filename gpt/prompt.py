import openai
import torch
import argparse



# Create arguments for the script
parser = argparse.ArgumentParser(description='What task do you want to run on GPT')
parser.add_argument('--task', type=str, default='rnn_identity_numerical', help='task name')


args = parser.parse_args()
task = args.task

def read_pt_file(filepath):
    # Assuming the .pt file contains data in a PyTorch tensor format
    data = torch.load(filepath)
    return data

def format_prompt_for_formula(data):
    train_data = data[0][:10].tolist()  # Assuming this is a list of lists
    labels = data[1][:10].tolist()   # Assuming this is a list of lists
    prompt = "Each row in the table below contains two lists.\n"
    prompt += "Please give me a formula for how to compute y as a function of element in x.\n"
    prompt += "list x        list y\n"
    prompt += "----------------------------\n"
    
    for xy, z in zip(train_data, labels):
        xy_str = ",".join(map(str, xy))
        z_str = ",".join(map(str, z))
        prompt += f"{xy_str}       {z_str}\n"
    return prompt

def format_prompt_for_code(data):
    train_data = data[0][:10].tolist()   # Assuming this is a list of lists
    labels = data[1][:10].tolist()   # Assuming this is a list of lists
    prompt = "Each row in the table below contains two lists.\n"
    prompt += "Please write a Python program that computes the list b from the list a.\n\n"
    prompt += "list x          list y\n"
    prompt += "----------------------------\n"
    for a, b in zip(train_data, labels):
        a_str = ",".join(map(str, a))
        b_str = ",".join(map(str, b))
        prompt += f"{a_str}       {b_str}\n"
    return prompt

def query_gpt(prompt, api_key):
    openai.api_key = api_key
    response = openai.ChatCompletion.create(
        model="gpt-4", 
        messages=[{"role": "system", "content": "You are a helpful assistant."}, 
                  {"role": "user", "content": prompt}]
    )
    return response

def save_conversation(prompt, response, filename):
    with open(filename, 'w') as file:
        file.write("Prompt:\n")
        file.write(prompt + "\n\n")
        file.write("GPT response:\n")
        file.write(response.choices[0].message['content'])

# Main execution
file_path = f"../tasks/{task}/data.pt"
api_key = "your_openai_api_key"  # Replace with your OpenAI API key
output_file_formula = "output_formula.txt"
output_file_code = "output_code.txt"

data = read_pt_file(file_path)
prompt_formula = format_prompt_for_formula(data)
print(prompt_formula)
prompt_code = format_prompt_for_code(data)
print(prompt_code)
# response_formula = query_gpt(prompt_formula, api_key)
# response_code = query_gpt(prompt_code, api_key)

# save_conversation(prompt_formula, response_formula, output_file_formula)
# save_conversation(prompt_code, response_code, output_file_code)

