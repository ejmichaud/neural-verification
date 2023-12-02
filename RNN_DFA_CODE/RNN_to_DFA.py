import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Run all experiments on either Autoregression on MNIST task or Rosenbrock minimization task.')
parser.add_argument('-d', type=str, dest='dataset_name', required=True)
parsed_args = parser.parse_args()
dataset_name = parsed_args.dataset_name
print(dataset_name)
from problems import Seq2SeqModel
from problems import AlgorithmicDatasets

dataset_manager = AlgorithmicDatasets()

# Define hyperparameters
hyperparameters = dataset_manager.get_dataset_hyperparameters(dataset_name)
dataset_size = hyperparameters["dataset_size"]
sequence_length = hyperparameters["sequence_length"]
learning_rate = hyperparameters["learning_rate"]
num_epochs = hyperparameters["num_epochs"]
batch_size = hyperparameters["batch_size"]

train_in, train_out, test_in, test_out = dataset_manager.get_dataset(dataset_name, dataset_size=dataset_size, sequence_length=sequence_length, train_ratio=0.8)
print(f"Train Input Shape: {train_in.shape}")
print(f"Train Output Shape: {train_out.shape}")
print(f"Test Input Shape: {test_in.shape}")
print(f"Test Output Shape: {test_out.shape}")

os.makedirs("serpinski_reproduction/", exist_ok=True)
os.makedirs("serpinski_reproduction_graphs/", exist_ok=True)
os.makedirs("serpinski_reproduction_graphs_no_dfa/", exist_ok=True)


class Ellipse:
    def __init__(self, center, covariance):
        self.center = center
        self.covariance = covariance  # xx^T
    def affine_forward(self, weight, bias):
        return Ellipse(weight.dot(self.center) + bias, weight.dot(self.covariance).dot(weight))  # Ax+b, Axx^TA^T
    def loosen_realign(self):
        return NotImplementedError

class Circle:
    def __init__(self, center, r, n_tokens):
        self.center = center
        self.r = r
        self.class_num = None
        self.separation = 0
        self.n_tokens = n_tokens
        self.parent = None
        self.destinations = [None for token in range(n_tokens)]
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
        return Circle(np.linalg.inv(weight).dot(self.center-bias), self.r/np.linalg.svd(weight)[1][0], self.n_tokens)
    def tighten_tanh_backward(self):
        min_activation = np.maximum(0, np.min(np.abs(self.center))-self.r)
        max_gradient = 1-min_activation**2
        return Circle(np.arctanh(np.clip(self.center, -1+1e-10, 1-1e-10)), self.r/max_gradient, self.n_tokens)
    def loosen_forward(self, weights, token):
        return self.affine_forward(weights["hidden2hidden.weight"].numpy(), weights["hidden2hidden.bias"].numpy()+weights["in2hidden.weight"].numpy()[:,token]).loosen_tanh_forward()
    def tighten_backward(self, weights, token):
        return self.tighten_tanh_backward().affine_forward(weights["hidden2hidden.weight"].numpy(), weights["hidden2hidden.bias"].numpy()+weights["in2hidden.weight"].numpy()[:,token])

class NFA:
    def __init__(self, n_tokens, weights):
        self.n_tokens = n_tokens
        self.states = []
        self.epsilon_transition_matrix = np.zeros([0, 0]).astype(bool)
        self.token_transition_matrix = np.zeros([n_tokens, 0, 0]).astype(bool)
        self.weights = weights
        self.n_classes = self.weights["hidden2out.bias"].numpy().shape[0]

            
        # The variables below encode the shapes of the convex regions of hidden space where one class is chosen over another
        out_weights = self.weights["hidden2out.weight"].numpy()
        out_biases = self.weights["hidden2out.bias"].numpy()
        self.diffs = out_weights[:,np.newaxis,:] - out_weights[np.newaxis,:,:]  # preferred output, other output, input
        self.bias_diffs = out_biases[:,np.newaxis] - out_biases[np.newaxis,:]  # preferred output, other output
        norms = np.sqrt(np.sum(self.diffs**2+np.eye(out_biases.shape[0])[:,:,np.newaxis], axis=2))  # preferred_output, other_output
        self.diffs = self.diffs / norms[:,:,np.newaxis]  # preferred output, other output, input
        self.bias_diffs = self.bias_diffs / norms  # preferred output, other output
        self.diffs = np.stack([np.concatenate([self.diffs[i,:i,:], self.diffs[i,i+1:,:]], axis=0) for i in range(self.diffs.shape[0])], axis=0)  # class, condition, channel
        self.bias_diffs = np.stack([np.concatenate([self.bias_diffs[i,:i], self.bias_diffs[i,i+1:]], axis=0) for i in range(self.bias_diffs.shape[0])], axis=0)  # class, condition

    def add(self, state):
        state.class_num = self.classify(state.center)
        state.separation = self.class_radius(state.center, state.class_num)
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
                forward_i = self.states[i].loosen_forward(self.weights, token)
                backward_n = self.states[n-1].tighten_backward(self.weights, token)
                self.token_transition_matrix[token,i,n-1] = self.states[n-1].contains(forward_i) or self.states[i].contains(backward_n)
                forward_n = self.states[n-1].loosen_forward(self.weights, token)
                backward_i = self.states[i].tighten_backward(self.weights, token)
                self.token_transition_matrix[token,n-1,i] = self.states[i].contains(forward_n) or self.states[n-1].contains(backward_i)

    def prune(self):
        pass

    def find_parent(self, point_or_region, excluded_state=None):
        for state in self.states:
            if state is not excluded_state and state.contains(point_or_region):
                return state
        return None

    def classify(self, point):
        return int(np.argmax(np.all(np.sum(self.diffs*point, axis=2) + self.bias_diffs > 0, axis=1)))

    def class_radius(self, point, point_class):
        return np.min(np.sum(self.diffs[point_class,:,:]*point, axis=1) + self.bias_diffs[point_class,:], axis=0)

    def accommodate_points(self, points, percent_to_boundary):
        classes = np.array([self.classify(point) for point in points])
        points_by_classes = [[] for class_ in range(self.n_classes)]
        for i in range(len(points)):
            points_by_classes[classes[i]].append(points[i])
        radii_by_classes = [[self.class_radius(point, class_num) for point in class_] for class_num, class_ in enumerate(points_by_classes)]
        argsort_indices = [np.argsort(radii) for radii in radii_by_classes]
        radii_by_classes = [[class_radii[index] for index in class_indices.tolist()] for class_radii, class_indices in zip(radii_by_classes, argsort_indices)]
        points_by_classes = [[class_points[index] for index in class_indices.tolist()] for class_points, class_indices in zip(points_by_classes, argsort_indices)]

        for class_num in range(self.n_classes):
            for ind in range(len(points_by_classes[class_num])):
                if not self.find_parent(points_by_classes[class_num][ind]):
                    self.add(Circle(points_by_classes[class_num][ind], radii_by_classes[class_num][ind]*percent_to_boundary, self.n_classes))

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
        unchecked_states = self.states[:]

        while unchecked_states:
            state = unchecked_states.pop(0)
            if state.parent:
                continue
            for token in range(self.n_tokens):
                if state.destinations[token] is not None:
                    continue
                destination = state.loosen_forward(self.weights, token)
                state.destinations[token] = self.find_parent(destination, excluded_state=destination)
                if state.destinations[token] is None:
                    state.destinations[token] = destination
                    self.add(destination)
                    unchecked_states.append(destination)
            if len(self.states) > 50:
                break
        print(len(self.states))

thresholds = [2, 1.5, 1, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0]
n_experiments = 100
for experiment_number in range(n_experiments):
    for threshold in thresholds:
        name = "serpinski_reproduction/" + dataset_name + "_" + str(experiment_number) + "_" + str(threshold) + "_model.pt"
        name = "models/" + dataset_name+"_model.pt"
        if os.path.isfile(name):
            # Initialize the model
            dataset_shapes = dataset_manager.get_dataset_shapes(dataset_name)
            input_size = dataset_shapes["input_size"]
            hidden_size = dataset_shapes["hidden_size"]
            output_size = dataset_shapes["output_size"]
            model = Seq2SeqModel(input_size, hidden_size, output_size)
            weights = torch.load(name)
            model.load_state_dict(weights)

            # Create a DataLoader for training data
            train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_in, train_out), batch_size=batch_size, shuffle=False)  # SHUFFLE=TRUE BREAKS THE IN/OUT CORRESPONDENCE
            test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_in, test_out), batch_size=batch_size, shuffle=False)

            model.eval()
            points = []
            for in_batch, target_batch in test_loader:
                with torch.no_grad():
                    hiddens, output = model(in_batch)
                    points = points + [hiddens[i,j,:].numpy() for i in range(hiddens.size(0)) for j in range(hiddens.size(1))]

            nfa = NFA(input_size, weights)
            print(points[:3])
            nfa.accommodate_points(points, 0.8)
            nfa.find_transitions()

            points = np.stack(points, axis=1)

            fig, ax = plt.subplots()
            nfa.draw(ax)
            ax.scatter(points[0], points[1], c="k", marker=".")
            plt.savefig("serpinski_reproduction_graphs/" + name[23:-3] + ".jpg")
            plt.close()

            fig, ax = plt.subplots()
            ax.scatter(points[0], points[1], c="k", marker=".")
            plt.savefig("serpinski_reproduction_graphs_no_dfa/" + name[23:-3] + ".jpg")
            plt.close()

