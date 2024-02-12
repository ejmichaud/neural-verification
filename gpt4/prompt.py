from openai import OpenAI
import os
import torch
import argparse
import re
import os

# Replace with your OpenAI API key
api_key = "your-key-here"
client = OpenAI(api_key=api_key)

# Create arguments for the script
parser = argparse.ArgumentParser(description='Run GPT tasks')
parser.add_argument('--task', type=str, default='rnn_identity_numerical', help='task name')
parser.add_argument('--iterations', type=int, default=1, help='number of iterations to run the task')
args = parser.parse_args()
task = args.task
iterations = args.iterations

def write_to_file(filename, content):
    with open(filename, 'w') as file:
        file.write(content)

def read_pt_file(filepath):
    # Assuming the .pt file contains data in a PyTorch tensor format
    data = torch.load(filepath)
    return data


def format_prompt_for_formula(data):
    train_data = data[0][:10].tolist()  # Assuming this is a list of lists
    labels = data[1][:10].tolist()  # Assuming this is a list of lists
    prompt = "Each row in the table below contains two lists.\n"
    prompt += "Please give me a formula for how to compute y as a function of elements in x.\n"
    prompt += "list x                       list y\n"
    prompt += "--------------------------------------------------------\n"
    
    for xy, z in zip(train_data, labels):
        xy_str = ",".join(map(str, xy))
        z_str = ",".join(map(str, z))
        prompt += f"{xy_str:<30} {z_str}\n"
    return prompt

def format_prompt_for_code():
    prompt = "Please write a Python program to compute list y from list x for the first 5 rows. " \
             "Check if the output matches list y and print 'Success' or 'Failure'.\n"
    return prompt


def extract_and_save_code_block(text, output_directory, iteration):
    # Define the regex pattern to match the code block
    start_delimiter = re.escape("```python")
    end_delimiter = re.escape("```")
    pattern = rf"{start_delimiter}(.*?){end_delimiter}"

    # Use regex to find the code block in the text
    match = re.search(pattern, text, re.DOTALL)
    if match:
        code_block = match.group(1).strip()  # Extract the code block and strip leading/trailing whitespace

        # Save the code block to a file with iteration number in the filename
        code_filename = os.path.join(output_directory, f"extracted_code_{iteration}.py")
        with open(code_filename, 'w') as code_file:
            code_file.write(code_block)
        print(f"Code block saved to {code_filename}")
    else:
        print("No matching code block found.")


def query_gpt(messages, api_key):
    formatted_messages = [{"role": msg.get("role"), "content": msg.get("content")} for msg in messages]
    response = client.chat.completions.create(
        model="gpt-4-1106-preview", 
        messages=formatted_messages
    )
    print(response)
    return response  # Return the entire response object

def save_conversation(messages, filename):
    with open(filename, 'w') as file:
        file.write("Conversation:\n")
        for message in messages:
            file.write(f"{message['role']}: {message['content']}\n")

# Main execution

for iteration in range(iterations):
    iteration_output_directory = f"./{task}/{iteration}"
    iteration_output_file = f"{iteration_output_directory}/output_conversation.txt"

    # Create the output directory for this iteration if it doesn't exist
    if not os.path.exists(iteration_output_directory):
        os.makedirs(iteration_output_directory)

    # Get the data for the prompts
    file_path = f"../tasks/{task}/data.pt"
    data = read_pt_file(file_path)

    # Reset the messages array
    messages = [{"role": "system", "content": "You are a helpful assistant."}]

    # Formatting and querying GPT for the formula
    prompt_formula = format_prompt_for_formula(data)
    messages.append({"role": "user", "content": prompt_formula})
    response = query_gpt(messages, api_key)
    if response.choices:
        assistant_message_content = response.choices[0].message.content  # Extract the string content
        messages.append({"role": "assistant", "content": assistant_message_content})

    # Formatting and querying GPT for the code
    prompt_code = format_prompt_for_code()
    messages.append({"role": "user", "content": prompt_code})
    response = query_gpt(messages, api_key)
    if response.choices:
        assistant_message_content = response.choices[0].message.content  # Extract the string content
        messages.append({"role": "assistant", "content": assistant_message_content})
        extract_and_save_code_block(assistant_message_content, iteration_output_directory, iteration)

    # Save the entire conversation for this iteration
    save_conversation(messages, iteration_output_file)

