import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import numpy as np
import os
import matplotlib.pyplot as plt
import yaml
import tqdm

from neural_verification import MLP, MLPConfig
from neural_verification import (
    GeneralRNNConfig,
    GeneralRNN,
    cycle,
    FastTensorDataLoader
)


#parser = argparse.ArgumentParser(description='Run all experiments on either Autoregression on MNIST task or Rosenbrock minimization task.')
#parser.add_argument('-d', type=str, dest='dataset_name', required=True)
#parsed_args = parser.parse_args()
#dataset_name = parsed_args.dataset_name


# Read search.yaml
with open("../search.yaml", 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

# Get a list of all the weights for tasks that are suitable for DFA conversion
tasks_with_networks = {}
for task, task_config in config.items():
    print(task, end=" ")

    if not task_config['include']:
        print("flagged as do not include")
        continue # skip this task

    # Pick the tasks that map sequences of tokens to sequences of tokens
    task_args = task_config['args']
    if task_args['vectorize_input'] == True and task_args['loss_fn'] == 'cross_entropy':
        if task_args['activation'] != 'Tanh':  # skip non-Tanh tasks
            print("not a Tanh network")
            continue
        task_path = "rnn_tests/raw_models/" + task + ".pt"
        if not os.path.exists(task_path):  # skip tasks that failed to train
            print("failed to train")
            continue

        print("loaded")
        if torch.cuda.is_available():
            tasks_with_networks[task] = torch.load(task_path)
        else:
            tasks_with_networks[task] = torch.load(task_path, map_location=torch.device('cpu'))
    else:
        print("not a token-in-token-out task")

print("To convert: ", tasks_with_networks.keys())

# Get all the RNN shapes
tasks_with_configs = {}
for task, weights in tasks_with_networks.items():
    hidden_mlp_depth = 0
    output_mlp_depth = 0
    for key in weights.keys():
        if key[:4] == 'hmlp' and key[-6:] == 'weight':
            hidden_mlp_depth += 1
        if key[:4] == 'ymlp' and key[-6:] == 'weight':
            output_mlp_depth += 1
    hidden_dim = weights['hmlp.mlp.' + str(hidden_mlp_depth*2-2) + '.weight'].shape[0]
    output_dim = weights['ymlp.mlp.' + str(output_mlp_depth*2-2) + '.weight'].shape[0]
    input_dim = weights['hmlp.mlp.0.weight'].shape[1] - hidden_dim
    if hidden_mlp_depth >= 2:
        hidden_mlp_width = weights['hmlp.mlp.0.weight'].shape[0]
    else:
        hidden_mlp_width = hidden_dim
    if output_mlp_depth >= 2:
        output_mlp_width = weights['ymlp.mlp.0.weight'].shape[0]
    else:
        output_mlp_width = output_dim
    activation = nn.Tanh

    config = GeneralRNNConfig(input_dim, output_dim, hidden_dim, hidden_mlp_depth, hidden_mlp_width, output_mlp_depth, output_mlp_width, activation)
    tasks_with_configs[task] = config


class GeneralRNNWithHiddenSaving(GeneralRNN):
    def forward_sequence(self, x):
        """This function takes in a sequence of inputs and returns a sequence of outputs
        as well as the final hidden state."""
        # x shape: (batch_size, sequence_length, input_dim)
        batch_size = x.size(0)
        seq_length = x.size(1)
        hidden = torch.zeros(batch_size, self.hidden_dim).to(self.device)
        assert x.size(2) == self.config.input_dim
        assert x.device == self.device
        hiddens = []
        outs = []
        for i in range(seq_length):
            out, hidden = self.forward(x[:,i,:], hidden)
            hiddens.append(hidden)
            outs.append(out)
        # out shape: (batch_size, sequence_length, output_dim)
        return torch.stack(outs).permute(1,0,2), torch.stack(hiddens).permute(1,0,2)

# Get all the RNN weights
tasks_with_rnns = {}
for task, weights in tasks_with_networks.items():
    rnn = GeneralRNNWithHiddenSaving(tasks_with_configs[task], device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    rnn.load_state_dict(weights)
    tasks_with_rnns[task] = rnn

# Get all the datasets
tasks_with_datasets = {}
for task in tasks_with_rnns.keys():
    if not os.path.exists(os.path.join("../tasks/", task, 'data.pt')):
        # run the create_dataset.py script
        os.system(f'python {os.path.join("../tasks/", task, "create_dataset.py")}')

    # data is loaded as x_train, y_train, x_test, y_test
    tasks_with_datasets[task] = tuple(torch.load(os.path.join("../tasks/", task, 'data.pt')))

class Circle:
    def __init__(self, center, r, n_tokens):
        self.center = center
        self.r = r
        self.class_num = None
        self.separation = 0
        self.n_tokens = n_tokens
        self.parent = None
        self.destinations = [None for token in range(n_tokens)]
    def copy(self):
        return Circle(self.center, self.r, self.n_tokens)
    def affine_forward(self, weight, bias):
        return Circle(weight.dot(self.center) + bias, self.r*np.linalg.svd(weight)[1][0], self.n_tokens)  # Ax+b, Axx^TA^T
    def loosen_realign(self):
        return Circle(self.center, self.r, self.n_tokens)
    def contains(self, other):
        if isinstance(other, Circle):
            return self.r >= np.sqrt(np.sum((self.center - other.center)**2)) + other.r
        elif isinstance(other, torch.Tensor) or isinstance(other, np.ndarray):
            return self.r >= np.sqrt(np.sum((self.center - other)**2))
        raise ValueError("Bad type: " + str(type(other)))
    def loosen_tanh_forward(self):
        min_preactivation = np.maximum(0, np.min(np.abs(self.center))-self.r)
        max_gradient = 1-np.tanh(min_preactivation)**2
        return Circle(np.tanh(self.center), max_gradient*self.r, self.n_tokens)
    def affine_backward(self, weight, bias):
        raise NotImplementedError
#        return Circle(np.linalg.inv(weight).dot(self.center-bias), self.r/np.linalg.svd(weight)[1][0], self.n_tokens)
    def tighten_tanh_backward(self):
        min_activation = np.maximum(0, np.min(np.abs(self.center))-self.r)
        max_gradient = 1-min_activation**2
        return Circle(np.arctanh(np.clip(self.center, -1+1e-10, 1-1e-10)), self.r/max_gradient, self.n_tokens)
    def loosen_forward(self, mlp, token=None):
        circle = self
        for index, layer in enumerate(mlp):
            if index == 0 and token is not None:
                circle = circle.affine_forward(layer.weight.numpy()[:,:circle.center.shape[0]], layer.bias.numpy()+layer.weight.numpy()[:,circle.center.shape[0]+token])
            elif index % 2 == 1:
                circle = circle.loosen_tanh_forward()
            else:
                circle = circle.affine_forward(layer.weight.numpy(), layer.bias.numpy())
        return circle
    def tighten_backward(self, weights, token):
        raise NotImplementedError  # requires affine_backward
#        circle = self
#        for index, layer in reversed(enumerate(mlp.mlp)):
#            if index == 0:
#                circle = circle.affine_backward(layer.weight.numpy()[:,:circle.r.shape[0]], layer.bias.numpy()+layer.weight.numpy()[:,circle.r.shape[0]+token])
#            elif index % 2 == 0:
#                circle = circle.tighten_tanh_backward()
#            else:
#                circle = circle.affine_backward(layer.weight.numpy(), layer.bias.numpy())
#        return circle

class NFA:
    def __init__(self, rnn):
        self.n_tokens = rnn.config.input_dim
        self.states = []
        self.epsilon_transition_matrix = np.zeros([0, 0]).astype(bool)
        self.token_transition_matrix = np.zeros([self.n_tokens, 0, 0]).astype(bool)
        self.rnn = rnn
        self.n_classes = rnn.config.output_dim

        # The variables below encode the shapes of the convex regions of hidden space where one class is chosen over another
        out_weights = self.rnn.ymlp.mlp[-1].weight.numpy()
        out_biases = self.rnn.ymlp.mlp[-1].bias.numpy()
#        out_weights = self.weights["hidden2out.weight"].numpy()
#        out_biases = self.weights["hidden2out.bias"].numpy()
        self.diffs = out_weights[:,np.newaxis,:] - out_weights[np.newaxis,:,:]  # preferred output, other output, input
        self.bias_diffs = out_biases[:,np.newaxis] - out_biases[np.newaxis,:]  # preferred output, other output
        norms = np.sqrt(np.sum(self.diffs**2+np.eye(out_biases.shape[0])[:,:,np.newaxis], axis=2))  # preferred_output, other_output
        self.diffs = self.diffs / norms[:,:,np.newaxis]  # preferred output, other output, input
        self.bias_diffs = self.bias_diffs / norms  # preferred output, other output
        self.diffs = np.stack([np.concatenate([self.diffs[i,:i,:], self.diffs[i,i+1:,:]], axis=0) for i in range(self.diffs.shape[0])], axis=0)  # class, condition, channel
        self.bias_diffs = np.stack([np.concatenate([self.bias_diffs[i,:i], self.bias_diffs[i,i+1:]], axis=0) for i in range(self.bias_diffs.shape[0])], axis=0)  # class, condition
        self.beheaded_ymlp = nn.Sequential(*list(self.rnn.ymlp.mlp.children())[:-1])

    def add(self, state):
        state.class_num = self.classify(state.center)
        state.separation = self.class_separation(state.center, state.class_num)
        state.parent = self.find_parent(state, excluded_state=state)
        self.states.append(state)
        self.epsilon_transition_matrix = np.concatenate([self.epsilon_transition_matrix, np.zeros([len(self.states)-1, 1], dtype=bool)], axis=1)
        self.epsilon_transition_matrix = np.concatenate([self.epsilon_transition_matrix, np.zeros([1, len(self.states)], dtype=bool)], axis=0)
        self.token_transition_matrix = np.concatenate([self.token_transition_matrix, np.zeros([self.n_tokens, len(self.states)-1, 1], dtype=bool)], axis=2)
        self.token_transition_matrix = np.concatenate([self.token_transition_matrix, np.zeros([self.n_tokens, 1, len(self.states)], dtype=bool)], axis=1)
        n = len(self.states)
        for i in range(n):
            self.epsilon_transition_matrix[i,n-1] = i==n-1 or self.states[n-1].contains(self.states[i])
            self.epsilon_transition_matrix[n-1,i] = i==n-1 or self.states[i].contains(self.states[n-1])
            for token in range(self.n_tokens):
                forward_i = self.states[i].loosen_forward(self.rnn.hmlp.mlp, token)
#                backward_n = self.states[n-1].tighten_backward(self.weights, token)
                self.token_transition_matrix[token,i,n-1] = self.states[n-1].contains(forward_i)# or self.states[i].contains(backward_n)
                forward_n = self.states[n-1].loosen_forward(self.rnn.hmlp.mlp, token)
#                backward_i = self.states[i].tighten_backward(self.weights, token)
                self.token_transition_matrix[token,n-1,i] = self.states[i].contains(forward_n)# or self.states[n-1].contains(backward_i)

    def prune(self):
        pass

    def find_parent(self, point_or_region, excluded_state=None):
        for state in self.states:
            if state is not excluded_state and state.contains(point_or_region):
                return state
        return None

    def classify(self, point):
        point = self.beheaded_ymlp(torch.tensor(point))
        return int(np.argmax(np.all(np.sum(self.diffs*point.numpy(), axis=2) + self.bias_diffs > 0, axis=1)))

    def class_separation(self, point, point_class):
        point = self.beheaded_ymlp(torch.tensor(point))
        return np.min(np.sum(self.diffs[point_class,:,:]*point.numpy(), axis=1) + self.bias_diffs[point_class,:], axis=0)

    def accommodate_points(self, points, percent_to_boundary):
        progress_bar = tqdm.tqdm(total=len(points), desc="Solving for radii", unit="point")
        classes = np.array([self.classify(point) for point in points])
        points_by_classes = [[] for class_ in range(self.n_classes)]
        for i in range(len(points)):
            points_by_classes[classes[i]].append(points[i])
        radii_by_classes = [[] for class_ in points_by_classes]
        for class_num, class_ in enumerate(points_by_classes):
            for point in class_:
                def sample_percent(r):
                    return Circle(point, r, 1).loosen_forward(self.beheaded_ymlp).r
                def binary_search(tolerance=1e-2):
                    r = self.class_separation(point, class_num)
                    high = r
                    low = r
                    target = r*percent_to_boundary
                    target_high = target*(1+tolerance)
                    target_low = target*(1-tolerance)
                    while sample_percent(high) < target_high:
                        high *= 2
                    while sample_percent(low) > target_low:
                        low /= 2
                    while True:
                        mid = (high+low)/2
                        sample = sample_percent(mid)
                        if target_low < sample and sample < target_high:
                            return mid
                        if sample_percent(mid) > target:
                            high = mid
                        else:
                            low = mid
                radii_by_classes[class_num].append(binary_search())
                progress_bar.update(1)
        progress_bar.close()
#        radii_by_classes = [[self.class_separation(point, class_num) for point in class_] for class_num, class_ in enumerate(points_by_classes)]
        argsort_indices = [np.argsort(radii) for radii in radii_by_classes]
        radii_by_classes = [[class_radii[index] for index in class_indices.tolist()] for class_radii, class_indices in zip(radii_by_classes, argsort_indices)]
        points_by_classes = [[class_points[index] for index in class_indices.tolist()] for class_points, class_indices in zip(points_by_classes, argsort_indices)]

        progress_bar = tqdm.tqdm(total=len(points), desc="Adding points", unit="point")
        for class_num in range(self.n_classes):
            for ind in range(len(points_by_classes[class_num])):
                if not self.find_parent(points_by_classes[class_num][ind]):
                    point = points_by_classes[class_num][ind]
                    radius = radii_by_classes[class_num][ind]*percent_to_boundary
                    self.add(Circle(point, radius, self.n_tokens))
                progress_bar.update(1)
        progress_bar.close()

    def draw(self, ax):
        assert self.diffs.shape[2] == 2
        class_colors = np.random.uniform(0, 1, size=(self.n_classes, 3))
        class_colors = np.eye(3).astype(np.float64)
        white = np.ones([3]).astype(np.float64)
        black = np.zeros([3]).astype(np.float64)
        grey = white / 2
        for class_num in range(self.n_classes):
            for condition_num in range(class_num):
                ax.axline(-self.diffs[class_num,condition_num,:]*self.bias_diffs[class_num,condition_num], slope=-self.diffs[class_num,condition_num,0]/self.diffs[class_num,condition_num,1], color="black", linestyle=(0, (5, 5)))
        successful = True
        for state in self.states:
            outstanding = not state.parent and any([dest is None for dest in state.destinations])
            classless = state.separation < state.r
            successful = successful and (not outstanding and not classless)
            class_num = state.class_num
            parent = state.parent
            class_color = class_colors[class_num,:]
            if outstanding:
                color = grey
                linestyle = "-"
            elif classless:
                color = black
                linestyle = "-"
            elif parent:
                color = (white + class_color) / 2
                linestyle = "--"
            else:
                color = class_color
                linestyle = "-"
            circle = plt.Circle(state.center, state.r, color=color, fill=False, linestyle=linestyle)
            ax.add_patch(circle)
            if outstanding or parent:
                continue
            for token in range(self.n_tokens):
                start = state.center
                end = state.destinations[token].center
                diff = end-start
                direction = diff / np.sqrt(np.sum(diff**2))
                perp_direction = np.array([[0, -1], [1, 0]]).dot(direction)
                start = start + direction*state.r + perp_direction*0.005
                if np.sqrt(np.sum(diff**2)) + state.destinations[token].r < state.r:
                    end = end + direction*state.destinations[token].r + perp_direction*0.01
                else:
                    end = end - direction*state.destinations[token].r + perp_direction*0.01
                ax.arrow(start[0], start[1], end[0]-start[0], end[1]-start[1], color=class_colors[token,:], length_includes_head=True, width=0.005, head_width=0.05, head_length=0.03, shape="right")
        if not successful:
            ax.plot((-1.2, 1.2), (-1.2, 1.2), "r")
            ax.plot((-1.2, 1.2), (1.2, -1.2), "r")
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)

    def find_transitions(self):

        n_states_limit = 200
        total_progress_bar = tqdm.tqdm(total=n_states_limit, desc="Total States", unit="states", position=0)
        verified_progress_bar = tqdm.tqdm(total=n_states_limit, desc="Verified States", unit="states", position=1)
        total_progress_bar.update(len(self.states))

        unchecked_states = self.states[:]
#        print(self.n_tokens)
        while unchecked_states:
            state = unchecked_states.pop(0)
            verified_progress_bar.update(1)
#            print(len(state.destinations))
            if state.parent:
                continue
            for token in range(self.n_tokens):
                if state.destinations[token] is not None:
                    continue
                destination = state.loosen_forward(self.rnn.hmlp.mlp, token)
                state.destinations[token] = self.find_parent(destination, excluded_state=destination)
                if state.destinations[token] is None:
                    state.destinations[token] = destination
                    self.add(destination)
                    unchecked_states.append(destination)
                    total_progress_bar.update(1)
                    if len(self.states) > n_states_limit:
                        break
            if len(self.states) > n_states_limit:
                break
        total_progress_bar.close()
        verified_progress_bar.close()
        print(len(self.states))
        if unchecked_states:
            return False
        return True




# Create all the hidden states
tasks = [
"rnn_dihedral",
#"rnn_permutations",
#"rnn_transpositions",
#"rnn_add_mod",
#"rnn_mult_mod",
#"rnn_div_mod",
#"rnn_sub_mod",
#"rnn_alternating4",
]

#for task in tasks_with_datasets.keys():
for task in tasks:
    batch_size = 8
    rnn = tasks_with_rnns[task]
    rnn.eval()
    points = []
    train_in = tasks_with_datasets[task][0]
    print(tasks_with_configs[task])
    in_batch = F.one_hot(train_in[:batch_size,:].type(torch.long), num_classes=rnn.config.input_dim).to(rnn.device)
    with torch.no_grad():
        outputs, hiddens = rnn.forward_sequence(in_batch)
        points = points + [hiddens[i,j,:].numpy() for i in range(hiddens.size(0)) for j in range(hiddens.size(1))]

        print("Constructing DFA for " + str(task))
        nfa = NFA(rnn)

        nfa.accommodate_points(points, 0.9)
        success = nfa.find_transitions()
        print(task, "successful dfa construction:", success)


