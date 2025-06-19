import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm
import os
import random
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from ibug import IBUGWrapper
from sklearn.utils import resample
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_squared_error
import logging
import time 
import yaml
import cohere
import seaborn as sns
import json
import glob
import re

# ignore warnings
import warnings
warnings.filterwarnings("ignore")



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm
import os
import random
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from ibug import IBUGWrapper
from sklearn.utils import resample
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.utils.data import TensorDataset, DataLoader
# ignore warnings
import warnings
warnings.filterwarnings("ignore")



class GPR:
    def __init__(self, kernel=None, alpha=1.0, random_state=42):
        self.alpha = alpha
        self.kernel = kernel if kernel else (
            C(1.0, (1e-5, 1e5)) * RBF(length_scale=1.0, length_scale_bounds=(1e-3, 1e3)) +
            WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-3, 1e6))
        )
        self.model = GaussianProcessRegressor(kernel=self.kernel, n_restarts_optimizer=10, random_state=random_state)

    def update_model(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def select_next_point(self, X_candidates):
        mean, std = self.model.predict(X_candidates, return_std=True)
        ucb = mean + self.alpha * std
        return np.argmax(ucb), ucb, mean, std

class RFR:
    def __init__(self,n_estimators=400, alpha=1.0, random_state=42):
        self.alpha = alpha
        self.model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)

    def update_model(self, X_train, y_train):
        self.model.fit(X_train, y_train)
    
    def rfPredictionIntervals(self, xVal, percentile=90):
        # initialize a list to hold the predictions from each tree
        y_preds = []
        # loop through the trees in the random forest
        for tree in self.model.estimators_:
            # get the predictions from each tree
            y_pred = tree.predict(xVal)
            # append the predictions to the list
            y_preds.append(y_pred)
        # Convert to np.array by stacking list of arrays along the column axis with each column being the prediction from a different tree
        y_preds = np.stack(y_preds, axis=1)           
        # get the quantiles for the confidence interval
        q_down = (100 - percentile) / 2.
        q_up = 100 - q_down

        # get the mean, uncertainty, lower bound, and upper bound
        y_lower = np.percentile(y_preds, q_down, axis=1)
        y_upper = np.percentile(y_preds, q_up, axis=1)  
        y_mean = self.model.predict(xVal)  
        y_uncert = y_upper - y_lower
        
        return y_mean, y_uncert


    def select_next_point(self, X_candidates):
        mean, uncertainty = self.rfPredictionIntervals(X_candidates)
        ucb = mean + self.alpha * uncertainty
        return np.argmax(ucb), ucb, mean, uncertainty

class XGB:
    def __init__(self, n_estimators=400, n_models=30, alpha = 1.0, random_state=42):
        self.alpha = alpha
        self.n_models = n_models
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.models = []

    def update_model(self, X_train, y_train):
        self.models = []
        rng = np.random.RandomState(self.random_state)
        for i in range(self.n_models):
            X_sample, y_sample = resample(X_train, y_train, random_state=rng.randint(0, 10000))
            model = XGBRegressor(
                n_estimators=self.n_estimators,
                reg_alpha=0,
                scale_pos_weight=1,
                base_score=0.5,
                random_state=rng.randint(0, 10000)
            )
            model.fit(X_sample, y_sample)
            self.models.append(model)

    def select_next_point(self, X_candidates):
        # Predict with all models
        y_preds = np.column_stack([model.predict(X_candidates) for model in self.models])

        # Compute uncertainty and mean
        y_mean = np.mean(y_preds, axis=1)
        y_std = np.std(y_preds, axis=1)
        ucb = y_mean + self.alpha * y_std
        return np.argmax(ucb), ucb, y_mean, y_std


#-------------------------------------------------------------------------------- BAYESIAN NEURAL NETWORK -------------------------------------------------------------------------------------------
class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Variational parameters for weights and biases
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-0.2, 0.2))
        self.weight_log_sigma = nn.Parameter(torch.Tensor(out_features, in_features).fill_(-5.0))

        self.bias_mu = nn.Parameter(torch.Tensor(out_features).uniform_(-0.2, 0.2))
        self.bias_log_sigma = nn.Parameter(torch.Tensor(out_features).fill_(-5.0))

        # Prior
        self.prior = Normal(0, 1)
        self.normal = Normal(0, 1)

    def forward(self, x):
        device = x.device

        weight_sigma = torch.exp(self.weight_log_sigma)
        bias_sigma = torch.exp(self.bias_log_sigma)

        # Create ε ~ N(0, 1) on the correct device
        weight_eps = torch.randn_like(self.weight_mu, device=device)
        bias_eps = torch.randn_like(self.bias_mu, device=device)

        weight = self.weight_mu + weight_sigma * weight_eps
        bias = self.bias_mu + bias_sigma * bias_eps

        return F.linear(x, weight, bias)


    def kl_loss(self):
        # Posterior: N(mu, sigma^2), Prior: N(0, 1)
        weight_sigma = torch.exp(self.weight_log_sigma)
        bias_sigma = torch.exp(self.bias_log_sigma)

        weight_kl = (torch.log(1.0 / weight_sigma) + (weight_sigma ** 2 + self.weight_mu ** 2 - 1) / 2).sum()
        bias_kl = (torch.log(1.0 / bias_sigma) + (bias_sigma ** 2 + self.bias_mu ** 2 - 1) / 2).sum()
        return weight_kl + bias_kl


class BayesianNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.layers = nn.ModuleList([
            BayesianLinear(input_dim, hidden_dim),
            BayesianLinear(hidden_dim, hidden_dim),
            BayesianLinear(hidden_dim, hidden_dim),
            BayesianLinear(hidden_dim, hidden_dim),
            BayesianLinear(hidden_dim, hidden_dim),
        ])
        self.out = BayesianLinear(hidden_dim, 1)

    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))
        return self.out(x)

    def kl_loss(self):
        return sum(layer.kl_loss() for layer in self.layers) + self.out.kl_loss()


def elbo_loss(predictions, targets, kl, beta=1.0):
    mse = F.mse_loss(predictions.squeeze(), targets, reduction='mean')
    return mse + beta * kl


def train_bnn(model, X_train, y_train, n_epochs=1000, lr=1e-3, beta=1.0, batch_size=64, device='cpu'):
    # Create DataLoader for batching
    dataset = TensorDataset(X_train, y_train)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.to(device)

    for epoch in range(n_epochs):
        model.train()
        total_loss = 0.0
        total_kl = 0.0
        total_batches = 0

        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            preds = model(X_batch)
            kl = model.kl_loss()
            loss = elbo_loss(preds, y_batch, kl, beta)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_kl += kl.item()
            total_batches += 1

        if epoch % 100 == 0:
            avg_loss = total_loss / total_batches
            avg_kl = total_kl / total_batches
            print(f"Epoch {epoch}, Avg Loss: {avg_loss:.4f}, Avg KL: {avg_kl:.2f}")


def predict_with_uncertainty(model, X, n_samples=1000):
    model.eval()
    #with torch.no_grad():
    preds = torch.stack([model(X).detach().squeeze() for _ in range(n_samples)])

    mean = preds.mean(0).cpu().numpy()
    std = preds.std(0).cpu().numpy()
    return mean, std

#-------------------------------------------------------------------------------- END CODE BAYESIAN NEURAL NETWORK -------------------------------------------------------------------------------------------


class BNN:
    def __init__(self, input_dim,device='cpu', alpha=1.0):
        self.alpha = alpha
        self.device=device
        self.model= BayesianNN(input_dim=input_dim).to(self.device)

    def update_model(self, X_train, y_train):
        X_train_tensor=torch.from_numpy(X_train).float().to(self.device)
        y_train_tensor=torch.from_numpy(y_train).float().to(self.device)
        train_bnn(self.model,X_train_tensor, y_train_tensor, batch_size=64,device=self.device)

    def select_next_point(self, X_candidates):
        X_candidates_tensor=torch.from_numpy(X_candidates).float().to(self.device)
        mean, std = predict_with_uncertainty(self.model,X_candidates_tensor)
        ucb = mean + self.alpha * std
        return np.argmax(ucb), ucb, mean, std





def run_MLAL(X_df, y_df, model_name, target_name, dataset_name, alpha=1.0, random_seed=42):
    X = X_df.values
    y = y_df.values
    max_target = y.max()

    num_col=X.shape[1]

    random.seed(random_seed)
    np.random.seed(random_seed)

    initial_idx = random.choice(list(range(len(X))))
    X_train = X[[initial_idx]]
    y_train = np.array([y[initial_idx]])

    if model_name == "GPR":
        model = GPR(alpha=alpha, random_state=random_seed)
    elif model_name == "RFR":
        model = RFR(alpha=alpha, random_state=random_seed)
    elif model_name == "XGB":
        model = XGB(alpha=alpha, random_state=random_seed)
    elif model_name == "BNN":
        model = BNN(num_col, alpha=alpha, device='cuda' if torch.cuda.is_available() else 'cpu')
    else:
        raise ValueError("Invalid model name. Choose from 'GPR', 'RFR', 'XGB', or 'BNN'.")
    

    # Indices of selected points
    selected = [initial_idx]
    # Observed values
    observed = [y[initial_idx]]
    # Mean predictions over all candidates
    mean_predictions = []
    # Uncertainty estimates
    uncertaintites = []

    iteration_indices = [0]
    trajectory_data = []

    for i, _ in enumerate(range(len(X) - 1)):
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        model.update_model(X_train_scaled, y_train)

        available = list(set(range(len(X))) - set(selected))

        X_candidates_scaled = scaler.transform(X[available])


        idx, ucb_scores, mean, uncertaintity = model.select_next_point(X_candidates_scaled)
        next_idx = available[idx]
        selected.append(next_idx)
        observed.append(y[next_idx])
        X_train = np.vstack([X_train, X[[next_idx]]])
        y_train= np.append(y_train, y[next_idx])

        mean_predictions.append(mean[idx])
        uncertaintites.append(uncertaintity[idx])

        iteration_indices.append(i+1)

        trajectory_data.append({
            "Iteration": i+1,
            "Index": next_idx,
            "Observed Target Value": y[next_idx],
            "Predicted Target Value": mean[idx],
            "Uncertainty": uncertaintity[idx],
            f"Max {target_name} in Dataset": max_target,
            "Stopping Reason": f"Max {target_name} reached" if y[next_idx] >= max_target else "Continuing"
        })

        if y[next_idx] >= max_target:
            print(f"Stopping early at iteration {i+1} - Max {target_name} found.")
            break

    df_traj = pd.DataFrame(trajectory_data)
    os.makedirs(f"al_trajectory_data_all/{dataset_name}", exist_ok=True)
    file_path = f"al_trajectory_data_all/{dataset_name}/{model_name}_trajectory_{dataset_name}_alpha{alpha}_seed{random_seed}.csv"
    df_traj.to_csv(file_path, index=False)
    print(f"Saved trajectory data to {file_path}")



data_grouped = pd.read_csv('steels_yield_report_featurized.csv')
X_df = data_grouped.drop(columns=['composition_original','yield strength','Report', 'Report with output', 'Formatted_Parameters'], errors='ignore')
#X_df = X_df.select_dtypes(include=[np.number]).iloc[:, 14:]
X_df = data_grouped.iloc[:, 2:16]
y_df = data_grouped['yield strength']


for model_name in ["GPR", "RFR", "XGB", "BNN"]:
    for alpha in [0, 0.1, 0.3, 0.5, 0.8, 1, 2, 3, 4, 5]:
        for seed in [42, 41, 40, 39, 38]:
            run_MLAL(X_df, y_df, model_name=model_name, target_name="Yield Strength", dataset_name="matbench_steels (composition)", alpha=alpha, random_seed=seed)



data_grouped = pd.read_csv('steels_yield_report_featurized.csv')
X_df = data_grouped.drop(columns=['composition_original','yield strength','Report', 'Report with output', 'Formatted_Parameters'], errors='ignore')
X_df = X_df.select_dtypes(include=[np.number]).iloc[:, 14:]
y_df = data_grouped['yield strength']


for model_name in ["GPR", "RFR", "XGB", "BNN"]:
    for alpha in [0, 0.1, 0.3, 0.5, 0.8, 1, 2, 3, 4, 5]:
        for seed in [42, 41, 40, 39, 38]:
            run_MLAL(X_df, y_df, model_name=model_name, target_name="Yield Strength", dataset_name="matbench_steels (featurized)", alpha=alpha, random_seed=seed)