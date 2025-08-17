# conda create -n nlp310 python=3.10 -y
# source /opt/conda/etc/profile.d/conda.sh
# conda activate nlp310
# 확인 : python -V
# pip install -r requirements.txt
# export TOKENIZERS_PARALLELISM=false
# python src/main.py --config configs/exp.yaml

# tmux attach -t exp
# docker exec -it hyoyoon-pretrain-nlp-exp1 bash
# conda activate test
# (conda install -y pytorch pytorch-cuda=12.1 -c pytorch -c nvidia) : torch 섪치
# (python -m pip install -U pip setuptools wheel)
# (pip install -r requirements.txt) ## 한번 설치하면 컨테이어 내 conda (test)에서 계속 유지
# wandb login
# python src/main.py --config configs/exp.yaml


# 로컬 터미널에서 tmux 
# ssh jeaheekim@147.47.39.138
# ssh jeaheekim@147.47.134.100 -p 2222
# 이동 : for_pretrain/hyoyoon/exp_1
# docker attach hyoyoon-pretrain-nlp-exp1 bash
# tmux attach -t exp 
# python src/main.py --config configs/exp.yaml


# watch? nvidia-smi


import wandb 
from tqdm import tqdm
import os

import torch
import torch.nn
import omegaconf
from omegaconf import OmegaConf

from .utils import load_config #,set_logger
from .model import EncoderForClassification
from .data import get_dataloader

# torch.cuda.set_per_process_memory_fraction(11/24) -> 김재희 로컬과 신입생 로컬의 vram 맞추기 용도. 과제 수행 시 삭제하셔도 됩니다. 
# model과 data에서 정의된 custom class 및 function을 import합니다.
"""
여기서 import 하시면 됩니다. 
"""
from transformers import set_seed


def train_iter(model, inputs, optimizer, device, epoch): 
    '''
    1 step 학습 (model.train() 상태에서 호출)
    '''
    inputs = {key : (value.to(device) if torch.is_tensor(value) else value) for key, value in inputs.items()} #
    outputs = model(**inputs)
    loss = outputs['loss']

    optimizer.zero_grad(set_to_none = True) ##
    loss.backward()
    
    optimizer.step()
    wandb.log({'train_loss' : loss.item(), 'train_epoch': epoch})
    return loss

def valid_iter(model, inputs, device):
    '''
    1 step 평가 (model.eval() & no_grad() 상태에서 호출)
    '''
    inputs = {key : (value.to(device) if torch.is_tensor(value) else value) for key, value in inputs.items()} #
    outputs = model(**inputs)
    loss = outputs['loss']
    accuracy = calculate_accuracy(outputs['logits'], inputs['label'])    
    return loss, accuracy

def calculate_accuracy(logits, label):
    preds = logits.argmax(dim=-1)
    correct = (preds == label).sum().item()
    return correct / label.size(0)

def main(configs : omegaconf.DictConfig) :
    # Set device
    set_seed(int(getattr(configs, 'seed', 42)))
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu') # 컨테이너 자체가 cuda:0 참조할 것

    ## wandb 로깅
    wandb.init(
        project=configs.logging.project,
        name=configs.logging.run_name,
        config=OmegaConf.to_container(configs, resolve=True)
    )

    # Load data
    train_loader = get_dataloader(configs.data, 'train')
    val_loader = get_dataloader(configs.data, 'valid')
    test_loader = get_dataloader(configs.data, 'test')

    # Load model
    model = EncoderForClassification(configs.model).to(device)

    # Set optimizer
    lr = float(getattr(configs.train_config, 'lr', 5e-5))
    weight_decay = float(getattr(configs.train_config, 'weight_decay', 0.0))
    optimizer = torch.optim.Adam(model.parameters(), # (p for p in model.parameters() if p.requires_grad)
                                 lr=lr, 
                                 weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _:1.0) ##


    # Train & validation for each epoch
    epochs = int(getattr(configs.train_config, 'epochs', 3))
    grad_clip = float(getattr(configs.train_config, "grad_clip", 0.0)) ##
    log_every = int(getattr(configs.train_config, "log_every", 50)) ##

    out_dir = getattr(configs.train_config, "output_dir", "outputs/exp1")
    os.makedirs(out_dir, exist_ok=True)
    best_ckpt = os.path.join(out_dir, "best.pt")
    best_val_acc = -1.0

    for epoch in range(1, epochs + 1):
        # ---- train ----
        model.train()
        running_loss, running_correct, running_total = 0.0, 0, 0
        pbar = tqdm(train_loader, desc=f"[train] epoch {epoch}")
        for step, batch in enumerate(pbar, start=1):
            loss = train_iter(model, batch, optimizer, device, epoch)

            # grad clip (옵션)
            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            scheduler.step()

            # 통계
            with torch.no_grad():
                batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
                logits = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    token_type_ids=batch.get("token_type_ids", None),
                    label=batch["label"],
                )["logits"]
                preds = logits.argmax(dim=-1)
                running_correct += (preds == batch["label"]).sum().item()
                bs = batch["label"].size(0)
                running_total += bs
                running_loss += loss.item() * bs

            if step % log_every == 0:
                wandb.log({
                    "train/acc": running_correct / max(running_total, 1),
                    "lr": optimizer.param_groups[0]["lr"],
                    "global_step": (epoch - 1) * len(train_loader) + step,
                })
                pbar.set_postfix(loss=f"{running_loss/max(running_total,1):.4f}",
                                 acc=f"{running_correct/max(running_total,1):.4f}")

        # ---- validation ----
        model.eval()
        total, correct, loss_sum = 0, 0, 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"[valid] epoch {epoch}", leave=False):
                loss, acc = valid_iter(model, batch, device)
                bs = batch["label"].size(0)
                loss_sum += loss.item() * bs
                correct += int(acc * bs)
                total += bs
        val_loss = loss_sum / max(total, 1)
        val_acc = correct / max(total, 1)

        wandb.log({"val/loss": val_loss, "val/acc": val_acc, "epoch": epoch})
        print(f"[epoch {epoch}] val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        # ---- save best ----
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({"model": model.state_dict(), "val_acc": val_acc, "epoch": epoch}, best_ckpt)
            print(f"   saved best: {best_ckpt} (val_acc={val_acc:.4f})")
    
    
    ## Test with best checkpoint ----
    ckpt = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="[test]"):
            loss, acc = valid_iter(model, batch, device)
            bs = batch["label"].size(0)
            loss_sum += loss.item() * bs
            correct += int(acc * bs)
            total += bs
    test_loss = loss_sum / max(total, 1)
    test_acc = correct / max(total, 1)

    wandb.log({"test/loss": test_loss, "test/acc": test_acc})
    print(f"[BEST] val_acc={best_val_acc:.4f} | [TEST] loss={test_loss:.4f} acc={test_acc:.4f}")

    # validation for last epoch
    
if __name__ == "__main__" :
    configs = load_config()
    main(configs)