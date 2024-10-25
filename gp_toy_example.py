import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split, Subset
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.generation import MaxPosteriorSampling
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.kernels import MaternKernel, ScaleKernel
from torch.quasirandom import SobolEngine
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
from tqdm import tqdm
import gpytorch
import copy
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float64

# Load datasets
wikitext = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
ag_news = load_dataset("ag_news", split="train")
imdb = load_dataset("imdb", split="train")
valid_wikitext = load_dataset("allenai/paloma", "falcon-refinedweb", split="val")

# Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
tokenizer.pad_token = tokenizer.eos_token

def preprocess(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

def shuffle_dataset(dataset):
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    return Subset(dataset, indices)

# Custom Dataset class
class TextDataset(Dataset):
    def __init__(self, dataset, preprocess_fn):
        self.dataset = dataset
        self.preprocess_fn = preprocess_fn

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        processed = self.preprocess_fn({"text": item["text"]})
        return {
            'input_ids': torch.tensor(processed['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(processed['attention_mask'], dtype=torch.long)
        }

wikitext_dataset = TextDataset(wikitext, preprocess)
ag_news_dataset = TextDataset(ag_news, preprocess)
imdb_dataset = TextDataset(imdb, preprocess)
validation_full = TextDataset(valid_wikitext, preprocess)
val_train_size = int(0.3 * len(validation_full))
val_train_dataset, val_test_dataset = random_split(validation_full, [val_train_size, len(validation_full) - val_train_size])
def eval_objective(x):
    """Evaluate the objective function (negative loss) for given weights"""
    weights = x.squeeze()
    datasets = {"wikitext": wikitext_dataset, "ag_news": ag_news_dataset, "imdb": imdb_dataset}
    datasets = {k: shuffle_dataset(v) for k, v in datasets.items()}
    batch_size = 32
    print("Weights to evaluate: ", weights)

    eval_model = copy.deepcopy(model)
    eval_model.to(device)
    eval_model.train()
    optimizer = optim.AdamW(eval_model.parameters(), lr=5e-5)

    for dataset_name, weight in zip(datasets, weights):
        dataset = datasets[dataset_name]
        n_samples = int(10000 * weight.item())
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        dataset_samples = 0
        print(dataset_name)
        for i, batch in enumerate(tqdm(dataloader)):
            if dataset_samples >= n_samples:
                break
            inputs = batch['input_ids'].to(device)
            optimizer.zero_grad()
            outputs = eval_model(inputs, labels=inputs)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            dataset_samples += inputs.size(0)

    eval_model.eval()
    val_dataloader = DataLoader(val_test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    total_loss = 0
    total_samples = 0
    with torch.no_grad():
        for batch in val_dataloader:
            inputs = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = eval_model(inputs, attention_mask=attention_mask, labels=inputs)
            loss = outputs.loss.item()
            total_loss += loss * inputs.size(0)
            total_samples += inputs.size(0)

    print("loss on val dataset: ", total_loss / total_samples)
    del eval_model
    torch.cuda.empty_cache()
    return -total_loss / total_samples  # Return negative loss as we want to maximize

def generate_batch(X, Y, batch_size, n_candidates):
    # Fit a GP model
    likelihood = GaussianLikelihood(noise_constraint=gpytorch.constraints.Interval(1e-8, 1e-3))
    covar_module = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=X.shape[-1]))
    model = SingleTaskGP(X, Y, likelihood=likelihood, covar_module=covar_module)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)

    # Draw samples on a Sobol sequence
    sobol = SobolEngine(X.shape[-1], scramble=True)
    X_cand = sobol.draw(n_candidates).to(dtype=dtype, device=device)

    # Thompson sample
    with torch.no_grad():
        thompson_sampling = MaxPosteriorSampling(model=model, replacement=False)
        X_next = thompson_sampling(X_cand, num_samples=batch_size)

    return X_next

def run_optimization(n_candidates, n_init, max_evals, batch_size):
    # Initial points
    sobol = SobolEngine(dimension=3, scramble=True)
    X = sobol.draw(n=n_init).to(dtype=dtype, device=device)
    X = X / X.sum(dim=1, keepdim=True)  # Normalize to sum to 1
    Y = torch.tensor([eval_objective(x) for x in X], dtype=dtype, device=device).unsqueeze(-1)
    print(f"{len(X)}) Best value: {Y.max().item():.2e}")

    while len(X) < max_evals:
        # Create a batch
        X_next = generate_batch(X, Y, batch_size=min(batch_size, max_evals - len(X)), n_candidates=n_candidates)
        X_next = X_next / X_next.sum(dim=1, keepdim=True)  # Normalize to sum to 1
        Y_next = torch.tensor([eval_objective(x) for x in X_next], dtype=dtype, device=device).unsqueeze(-1)

        # Append data
        X = torch.cat((X, X_next), dim=0)
        Y = torch.cat((Y, Y_next), dim=0)

        print(f"{len(X)}) Best value: {Y.max().item():.2e}")
    return X, Y

# Run optimization
n_candidates = 1000
n_init = 5
max_evals = 30
batch_size = 2

X_final, Y_final = run_optimization(n_candidates, n_init, max_evals, batch_size)

# Print best result
best_idx = Y_final.argmax()
best_weights = X_final[best_idx]
print(f"Best weights: WikiText: {best_weights[0]:.3f}, AG News: {best_weights[1]:.3f}, IMDB: {best_weights[2]:.3f}")
print(f"Best performance: {Y_final.max().item():.2e}")
