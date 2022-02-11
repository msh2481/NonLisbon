import torch
import torch.nn as nn
import numpy as np
import plotly.express as px
from tqdm import trange, tqdm
import neptune.new as neptune
from neptune.new.types import File

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_params(model):
    params = []
    treshold = 100
    for e in model.parameters():
        flat = e.view(-1)
        params.append(flat[:min(len(flat), treshold)].cpu().detach())
    return torch.cat(params, dim=0)

def eval_unity_root(poly, arg):
    num = np.exp(1j * arg)
    return np.polyval(poly, num)

def complex_hash(model, n):
    params = get_params(model)
    return np.abs(eval_unity_root(params, np.linspace(0, 2 * np.pi, num = n, endpoint = False)))

def test(model, dataloader):
    model.eval()
    loss_fn = nn.NLLLoss()
    test_loss, test_acc = 0, 0
    for x, y in dataloader:
        x, y = x.to(device), y.to(device, dtype=int)
        p = model(x)
        test_loss += loss_fn(p, y).item()
        test_acc += (p.argmax(dim=-1) == y).float().mean().item()
    return test_loss / len(dataloader), test_acc / len(dataloader)

def train_epoch(model, dataloader, optimizer, logging, plot_interval, augment = lambda x: x):
    if dataloader is None:
        return
    model.train()
    loss_fn = nn.NLLLoss()
    train_loss, train_acc = 0, 0
    for batch, (x, y) in enumerate(dataloader):
        optimizer.zero_grad()
        x, y = x.to(device), y.to(device, dtype=int)
        p = model(augment(x))
        loss = loss_fn(outputs, y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        train_acc += (p.argmax(dim=-1) == y).float().mean()
        if (batch + 1) % plot_interval == 0:
            logging(batch, train_loss / plot_interval, train_acc / plot_interval, *complex_hash(model, 2))

def train_model(trial, model, optimizer, scheduler, config):
    pathx, pathy = [], []
    min_loss = 1e9
    run_id = hex(int(time()))[2:]
    print(f'started train #{run_id}', flush=True)
    for epoch in trange(config['epochs']):
        def train_logging(batch, loss, acc, hx, hy):
            pathx.append(hx)
            pathy.append(hy)
            step = epoch + (batch + 1) / len(st.train_loader)
            if config['neptune_logging']:
                st.run['train/epoch'].log(step, step=step)
                st.run['train/train_loss'].log(loss, step=step)
                st.run['train/train_acc'].log(acc, step=step)
                st.run['train/path'] = File.as_html(px.line(x=pathx, y=pathy))
        def test_logging(loss, acc):
            nonlocal min_loss
            step = epoch + 1
            if config['neptune_logging']:
                st.run['train/epoch'].log(step, step=step)
                st.run['train/val_loss'].log(loss, step=step)
                st.run['train/val_acc'].log(acc, step=step)
            print(f'step: {step}, loss: {loss}, acc: {acc}, hx: {pathx[-1] if pathx else -1}, hy: {pathy[-1] if pathy else -1}')
            min_loss = min(min_loss, loss)
            trial.report(min_loss, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
            name = f'{run_id}_{epoch}'
            save_to_zoo(model, name, loss, acc)
        if config['use_per']:
            train_epoch_with_per(model, st.train_loader, optimizer, len(train_loader) // config['batch_size'], config['batch_size'], train_logging, config['plot_interval'])
        else:
            train_epoch_without_per(model, st.train_loader, optimizer, train_logging, config['plot_interval'])
        scheduler.step()
        with torch.no_grad():
            test_logging(*test(model, st.val_loader))
    return min_loss

from data import Plug, build_dataset, build_model, build_optimizer, build_lr_scheduler
from time import time
from torch.utils.data import DataLoader
from plotly import express as px
from autoaug import build_transforms
import neptune.new as neptune

def connect_neptune(project_name, run_token):
    st.project = neptune.init_project(name='mlxa/CNN', api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI5NTIzY2UxZC1jMjI5LTRlYTQtYjQ0Yi1kM2JhMGU1NDllYTIifQ==')
    st.run = neptune.init(project=project_name, api_token=run_token) if run_token else Plug()

def run(trial, config):
    [print(f'{key}: {value}', flush=True) for key, value in config.items()]
    st.device = config['device']
    st.run['parameters'] = config

    model = build_model(config)
    optimizer = build_optimizer(model.parameters(), config)
    train_set = build_dataset(config['train'])
    st.aug = build_transforms(config)
    # pics = st.aug(torch.stack([e[0] for e in train_set[:5]]))
    # for pic in pics:
    #     pic = pic.clip(0, 1)
    #     px.imshow(pic.permute(1, 2, 0)).show()
    if config['use_per']:
        if train_set:
            st.train_loader = PrioritizedReplayBuffer(size=len(train_set), alpha=config['per_alpha'], beta=config['per_beta'])
            for x, y in train_set:
                st.train_loader.add(Batch(obs=x, act=y, rew=0, done=False))
        else:
            st.train_loader = None
    else:
        st.train_loader = DataLoader(train_set, batch_size=config['batch_size'], shuffle=True) if train_set else None
    scheduler = build_lr_scheduler(optimizer, config)
    val_set, test_set = build_dataset(config['val']), build_dataset(config['test'])
    st.val_loader = DataLoader(val_set, batch_size=config['batch_size'], shuffle=False) if val_set else None
    st.test_loader = DataLoader(test_set, batch_size=config['batch_size'], shuffle=False) if test_set else None
    result = train_model(trial, model, optimizer, scheduler, config) if train_set else None
    solve_test(model, st.test_loader, f'solution_{model.loader}_{st.run_id}')
    return result