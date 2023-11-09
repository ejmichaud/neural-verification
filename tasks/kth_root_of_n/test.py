import torch


from neural_verification import (
    Tokenizer, 
    TransformerConfig, 
    Transformer, 
    cycle, 
    HeterogeneousDataLoader
)
from config import tokenizer_vocabulary, model_config
from train import compute_accuracy
device = "cpu"
tokenizer = Tokenizer(tokenizer_vocabulary)

model = Transformer(model_config).to(device)
model.load_state_dict(torch.load('model.pt', map_location=torch.device('cpu')))
model.eval()
input = "50,2|7"
input_ids = tokenizer.encode(input, padding=True)
print(input_ids, len(input_ids))
input_ids = torch.tensor(input_ids).reshape((1, len(input_ids))).to(device)
print(input_ids, input_ids.shape)

output = model(input_ids)
print(compute_accuracy(input_ids, output, [(4,)]))
output = output.argmax(dim=-1)
print(output, output.shape)
print(tokenizer.decode(output[0]))


input = "100,2|10"
input_ids = tokenizer.encode(input, padding=True)
print(input_ids, len(input_ids))
input_ids = torch.tensor(input_ids).reshape((1, len(input_ids))).to(device)
print(input_ids, input_ids.shape)

output = model(input_ids)
print(compute_accuracy(input_ids, output, [(4,)]))
output = output.argmax(dim=-1)
print(output, output.shape)
print(tokenizer.decode(output[0]))

input = "9993,2|100"
input_ids = tokenizer.encode(input, padding=True)
print(input_ids, len(input_ids))
input_ids = torch.tensor(input_ids).reshape((1, len(input_ids))).to(device)
print(input_ids, input_ids.shape)

output = model(input_ids)
output = output.argmax(dim=-1)
print(output, output.shape)
print(tokenizer.decode(output[0]))
# print(compute_accuracy(input_ids, output, [(4,)]))
