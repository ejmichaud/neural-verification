MLP.ipynb: defines a multilayer perceptron.

RNN.ipynb: defines an RNN. 

Transformer_token.ipynb: defines a transformer taking in token ids. For example, "She has a cat." => ["She", "has", "a", "cat"] => [0, 1, 2, 3]

Transformer_number.ipynb: defines a transformer taking in numbers. For example, multiplying two numbers.

Disclaimer: 
* These models may have many variants. The implementations in this folder are what Ziming finds widely used and are as simple as possible. It is likely that for some tasks, you need to change some details in the  model to make it work or interpretable.
* Don't be afraid of large language models, they are just large transformers! poly.ipynb shows how to train a transformer (defined in Transformer_token.ipynb) to do symbolic differentiation of polynomials. For example, the derivative of $2\*x\*\*2-3\*x+5$ is $4\*x-3$, then the sequence is ['2', '*', 'x', '\*\*', '2', '-', '3', '\*', 'x', '+', '5', '?', '4', '\*', 'x', '-', '3', '.', 'pad', 'pad',...] (pads are used to make all sentences to have equal lengths)

