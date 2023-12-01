import torch
from tqdm import tqdm
import os
import wandb
import numpy as np

wandb.init(project="pushdown-lm", entity="shikharmurty")
from transformers import (
    get_cosine_schedule_with_warmup,
)
from torch.optim import Adam, AdamW

from transformers.data.data_collator import DataCollatorWithPadding
from torch.utils.data import (
    DataLoader,
    RandomSampler,
    SequentialSampler,
)

import collate
from plot import CustomPlot


def get_grad_norm(model):
    total_norm = 0
    parameters = [
        p for p in model.parameters() if p.grad is not None and p.requires_grad
    ]

    for p in parameters:
        if len(p.shape) == 0:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm**0.5
    return total_norm


def get_opt(lr, model):
    if type(model) != torch.nn.Module:
        model = model.model
    no_decay = ["bias", "LayerNorm.weight"]
    weight_decay = 0.0
    adam_epsilon = 1e-7
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = Adam(
        optimizer_grouped_parameters,
        lr=lr,
        eps=adam_epsilon,
    )
    return optimizer


def get_scheduler(opt, t_total):
    num_warmup_steps = 8000
    scheduler = get_cosine_schedule_with_warmup(
        opt, num_warmup_steps=num_warmup_steps, num_training_steps=t_total
    )
    return scheduler


def eval_lm(model_interface, val_datasets, best_accs, device, num_steps, collator):
    def helper(validation):
        model_interface.model.eval()
        loss_curr = 0
        total = 0

        attachment_acc = 0.0
        stack_total = 0.0
        with torch.no_grad():
            for batch in tqdm(validation):
                batch_gpu = {}
                for key in batch:
                    batch_gpu[key] = batch[key].to(device)
                res = model_interface(batch_gpu, normalize=False)
                loss_curr += res["lm_out"].loss.cpu().numpy()
                total += (
                    (1 + batch_gpu["in_len"]).sum().item()
                )  ## sum all tokens, including the last word => </s> contrib

                attachment_acc_curr, stack_total_curr = res["attachment_acc"]
                attachment_acc += attachment_acc_curr
                stack_total += stack_total_curr

        if stack_total != 0.0:
            attachment_acc /= stack_total
        return np.exp(loss_curr / total), attachment_acc

    eval_batch_size =16
    plots = {}
    curr_accs = {}
    for key, val_dataset in val_datasets.items():
        validation = DataLoader(
            val_dataset,
            sampler=SequentialSampler(val_dataset),
            batch_size=eval_batch_size,
            collate_fn=collator,
        )
        curr_ppl, attachment_acc = helper(validation)
        curr_accs[key] = curr_ppl
        plots["curr-{}-attachment_acc".format(key)] = attachment_acc
        plots["curr-{}-ppl".format(key)] = curr_accs[key]

    best_accs = {key: min(curr_accs[key], best_accs[key]) for key in curr_accs}
    plots.update({"best/{}": v for k, v in best_accs.items()})
    plotting_util(plots, num_steps)
    return best_accs, curr_accs





def plotting_util(dict_of_elems, step):
    wandbdict = {}
    for k, v in dict_of_elems.items():
        if isinstance(v, CustomPlot):
            v = v.to_wandb()
            if v is None:
                continue

            if isinstance(v, dict):
                for k2, v2 in v.items():
                    wandbdict[k + "/" + k2] = v2
            else:
                wandbdict[k] = v
        elif isinstance(v, (int, float)):
            wandbdict[k] = v
        else:
            assert False, f"Invalid data type {type(v)}"
    wandbdict["iteration"] = step
    wandb.log(wandbdict)



def eval_func(model, validation, tokenizer, best_acc, device):
    def get_decoding_acc(outputs, labels):
        acc = 0
        for out, label in zip(outputs, labels):
            dec_str = tokenizer.decode(out, skip_special_tokens=True)
            label = [(l if l != -100 else tokenizer.pad_token_id) for l in label]
            orig_str = tokenizer.decode(label, skip_special_tokens=True)
            acc += dec_str == orig_str
        return acc

    curr_acc = 0
    total = 0
    if type(model) != torch.nn.Module:
        model.model.eval()
    else:
        model.eval()
    with torch.no_grad():
        for batch in tqdm(validation):
            batch_gpu = {}
            for key in batch:
                batch_gpu[key] = batch[key].to(device)
            curr_acc += get_decoding_acc(
                model.generate(batch_gpu["input_ids"]).cpu().tolist(),
                batch["labels"].cpu().tolist(),
            )
            total += len(batch["labels"])

    curr_acc /= 1.0 * total
    print("Current Accuracy: {:.4f}".format(curr_acc))
    if curr_acc > best_acc:
        return curr_acc
    else:
        return best_acc

def save_callback(model, optimizer, scheduler, save_dir, num_steps):
    state = {
        "step": num_steps,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
    }
    torch.save(
        state,
        os.path.join(save_dir, "full_state_{}.pickle".format(num_steps)),
    )


def train_loop(
    args,
    model,
    train_dataset,
    val_datasets,
    device,
    save_dir,
    tokenizer=None,
    callback_fn=None,
    skip_to_step=-1,
):
    num_steps = 0
    max_grad_norm = 3
    train_batch_size = 8
    accum_steps = 8
    eval_every = 3000
    max_steps = 300000

    opt = get_opt(args.lr, model)
    scheduler = get_scheduler(opt, max_steps)

    if args.get("resume_from_path", None):
        print("Resuming training from checkpoint: {}".format(args["resume_from_path"]))
        ### load optimizer and scheduler state
        checkpoint = torch.load(args["resume_from_path"])
        opt.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        model.model.load_state_dict(checkpoint["state_dict"])
        skip_to_step = checkpoint["step"]

    if tokenizer is not None:
        train_data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    elif model.model.mode in ["enc_dec", "lm"]:
        train_data_collator = collate.VarLengthCollate(tokenizer)

    best_accs = {key: 10000.0 for key in val_datasets}
    while True:
        train_dataloader = DataLoader(
            train_dataset,
            sampler=RandomSampler(train_dataset),
            batch_size=train_batch_size,
            collate_fn=train_data_collator,
        )
        total_train_sz = len(train_dataset)
        if num_steps > max_steps:
            break
        with torch.enable_grad(), tqdm(total=total_train_sz) as progress_bar:
            losses = []
            attachment_losses = []
            for curr_batch_dict in train_dataloader:
                if skip_to_step != -1:
                    if num_steps != skip_to_step:
                        num_steps += 1
                        progress_bar.update(curr_batch_dict["in"].shape[1])
                        continue
                    elif num_steps == skip_to_step:
                        print("Reached step!")
                        skip_to_step = -1

                if type(model) != torch.nn.Module:
                    model.model.train()
                else:
                    model.train()
                curr_batch_dict_gpu = {}
                for key in curr_batch_dict:
                    curr_batch_dict_gpu[key] = curr_batch_dict[key].to(device)

                out = model(curr_batch_dict_gpu)
                # Multi-task learn language modeling and attaching tokens
                loss_curr = out["lm_out"].loss + out["attachment_loss"]
                progress_bar.update(curr_batch_dict["in"].shape[1])
                losses.append(out["lm_out"].loss.item())
                if type(out["attachment_loss"]) != float:
                    attachment_losses.append(out["attachment_loss"].item())

                loss_curr /= accum_steps
                loss_curr.backward()
                if len(losses) == accum_steps:
                    grad_norm = get_grad_norm(model.model)
                    torch.nn.utils.clip_grad_norm_(
                        model.model.parameters(), max_grad_norm
                    )
                    log_dict = {
                        "lm_loss": sum(losses) / len(losses),
                        "iteration": num_steps,
                        "norm": grad_norm,
                    }
                    if sum(attachment_losses) != 0:
                        log_dict["attachment_loss"] = sum(attachment_losses) / len(
                            attachment_losses
                        )
                    progress_bar.set_postfix(log_dict)
                    wandb.log(log_dict)
                    opt.step()
                    scheduler.step()
                    model.model.zero_grad()
                    losses = []
                    attachment_losses = []
                    if num_steps % eval_every == 0:
                        print("Evaluating at step {}".format(num_steps))
                        if callback_fn is not None:
                            model.model.eval()
                            val_score = callback_fn("val")
                            test_score = callback_fn("test")
                            print(val_score)
                            print(test_score)
                            wandbdict = {
                                "iteration": num_steps,
                                "val_aux": val_score,
                                "test_aux": test_score,
                            }
                            wandb.log(wandbdict)
                        else:
                            best_accs, curr_accs = eval_lm(
                                model,
                                val_datasets,
                                best_accs,
                                device,
                                num_steps,
                                train_data_collator,
                            )
                            print(curr_accs)

                        if len(save_dir) > 0:
                            save_callback(
                                model.model, opt, scheduler, save_dir, num_steps
                            )
                    num_steps += 1
                    if num_steps > max_steps:
                        break
            if losses:
                grad_norm = get_grad_norm(model.model)
                log_dict = {
                    "lm_loss": sum(losses) / len(losses),
                    "iteration": num_steps,
                    "norm": grad_norm,
                }
                if sum(attachment_losses) != 0:
                    log_dict["attachment_loss"] = sum(attachment_losses) / len(
                        attachment_losses
                    )
                progress_bar.set_postfix(log_dict)
                wandb.log(log_dict)
                torch.nn.utils.clip_grad_norm_(model.model.parameters(), max_grad_norm)
                opt.step()
                scheduler.step()
                model.model.zero_grad()
                losses = []
                attachment_losses = []
                if num_steps % eval_every == 0:
                    print("Evaluating at step {}".format(num_steps))
                    if callback_fn is not None:
                        model.model.eval()
                        val_score = callback_fn("val")
                        test_score = callback_fn("test")
                        print(val_score)
                        print(test_score)
                        wandbdict = {
                            "iteration": num_steps,
                            "val_aux": val_score,
                            "test_aux": test_score,
                        }
                        wandb.log(wandbdict)
                    else:
                        best_accs, curr_accs = eval_lm(
                            model,
                            val_datasets,
                            best_accs,
                            device,
                            num_steps,
                            train_data_collator,
                        )
                        print(curr_accs)

                    if len(save_dir) > 0:
                        save_callback(model.model, opt, scheduler, save_dir, num_steps)
                num_steps += 1
                if num_steps > max_steps:
                    break

    print("Best Accuracies,", best_accs)
    return
