import json
import matplotlib.pyplot as plt
from typing import Any, Dict, List, Optional, Tuple, Union, Type
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from tqdm.auto import tqdm
from utils.metrics import pixel_accuracy, mean_IU, boundary_iou_after, computeQualityMeasures

def merge_dicts(list):
    output = {}
    for dict in list:
        for key, value in dict.items():
            if torch.is_tensor(value):
                value = value.to("cpu")
            if key not in output.keys():
                output.update({key: [value]})
            else:
                output[key].append(value)
    for key, value in output.items():
        output[key] = np.array(value)
    return output

class UnetTrainer:
    def __init__(self, models: dict,
                 loss_func: dict,
                 optimizers: dict,
                 batch_size: int,
                 log_path: str,
                 resume_training: bool = True,
                 hparameters=None,
                 redo_metric=True,
                 retest=False,
                 early_stopping=15
                 ):
        self._hparameter_list = ['seed', 'max_epoch', 'metric_best_epoch']
        # todo: which loss to use to evaluate
        self.best_value_metric = "boundary_iou"
        self.pick_best_fn = np.argmax
        # deal with cross-validation
        self._model_names = ['unet']
        self._opt_names = ['unet']
        self._loss_func_names = ['dice_loss', 'criterion']
        self.batch_size = batch_size
        self.log_dir = log_path
        self.models = models
        self.loss_func = loss_func
        self.optimizers = optimizers
        self.logger = SummaryWriter(self.log_dir)
        self.append_hparam(hparameters.keys())
        self.hparameters = hparameters
        self.resume_training = resume_training
        self.redo_metric = redo_metric
        self.retest = retest
        self.early_stopping_epoch = early_stopping

    def append_hparam(self, hps):
        for hp in hps:
            if hp not in self._hparameter_list:
                self._hparameter_list.append(hp)

    def reset_network_weights(self):
        for model in self.policy.models.values():
            for layer in model.children():
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()
        print("weights reset for policy networks")

    def is_training_done(self):
        return self.hparameters["trained_epoch"] == self.hparameters["max_epoch"]

    def start_training_log(self, resume_training, device):
        self.hparameters["metric"] = None
        self.hparameters["metric_best"] = {}
        self.hparameters["metric_best_epoch"] = None
        self.hparameters["trained_epoch"] = None
        self.hparameters["done"] = None
        need_to_train = None
        if not os.path.exists(os.path.join(self.log_dir, "training_log.json")) or not resume_training:
            p = {key: self.hparameters[key] for key in self._hparameter_list}
            print(f"training {p} from scratch")
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir, exist_ok=True)
            self.hparameters["trained_epoch"] = 0
            self.hparameters["metric"] = {}
            need_to_train = True

        else:
            hparameters_new = self.load_training_log(self.log_dir)
            # validate log
            if not set(self.hparameters.keys()) == set(hparameters_new.keys()):
                raise ValueError("unrecognized hparameter type")
            for key, value in self.hparameters.items():
                if key not in ["max_epoch", "metric", "trained_epoch", "done", "fold", "metric_best",
                               "metric_best_epoch"] \
                        and value != hparameters_new[key]:
                    raise ValueError(
                        f"input hyperparameter {key}:{value} mismatch with the existing one {hparameters_new[key]}")
            hparameters_new["done"] = self.hparameters["max_epoch"] <= hparameters_new["trained_epoch"]
            hparameters_new["max_epoch"] = self.hparameters["max_epoch"]
            self.hparameters = hparameters_new

            # training done
            if self.hparameters["done"]:
                print(f"already done {[self.hparameters[key] for key in self._hparameter_list]}, skip")
                need_to_train = False
            else:
                if self.hparameters["trained_epoch"] != len(self.hparameters["metric"][self.best_value_metric]):
                    raise ValueError(f"found {self.hparameters['trained_epoch']} trained epochs but "
                                     f"{len(self.hparameters['metric'][self.best_value_metric])} in metric")
                # validate model existence and load models
                ckpt_path = os.path.join(self.log_dir, f"ckpt-epoch{self.hparameters['trained_epoch']}.pth")
                self.load_checkpoints(ckpt_path, device)
                need_to_train = True
                p = {key: self.hparameters[key] for key in self._hparameter_list}
                print(f"resume training {p}")
        assert need_to_train is not None
        return need_to_train

    @staticmethod
    def load_training_log(log_path):
        with open(os.path.join(log_path, "training_log.json"), 'r') as fp:
            log = json.load(fp)
        return log

    @staticmethod
    def save_training_log(log_path, dict):
        with open(os.path.join(log_path, "training_log.json"), 'w') as fp:
            json.dump(dict, fp)
    
    def save_checkpoints(self, checkpoint_path):
        save_dict = {}
        for model_name in self._model_names:
            save_dict.update({model_name: self.models[model_name].state_dict()})
        for opt_name in self._opt_names:
            save_dict.update({f"optim_{opt_name}": self.optimizers[opt_name].state_dict()})
        torch.save(save_dict, checkpoint_path)

    def load_checkpoints(self, checkpoint_path, device):
        ckpt = torch.load(checkpoint_path, map_location=torch.device(device))
        for model_name in self._model_names:
            self.models[model_name] = self.models[model_name].to(device)
            self.models[model_name].load_state_dict(ckpt[model_name])
        for opt_name in self._opt_names:
            self.optimizers[opt_name].load_state_dict(ckpt[f"optim_{opt_name}"])


    # differ in training times?
    def train_epoch(self, train_loader, epoch, log_dir, device):
        for name in self._model_names:
            self.models[name].train()

        training_loss = {'unet_loss':[]}
      
        for dic, target, dist_map, _ in tqdm(train_loader, desc=f"training epoch #{epoch}"):

            self.optimizers['unet'].zero_grad()
            output = self.models['unet'](dic.to(device))
            #boundary loss
            #l2_loss = self.loss_func['l2'](self.models['unet'], 0.001)
            pred_probs = F.softmax(output, dim=1)
            gdl_loss = self.loss_func['dice_loss'](pred_probs.to('cpu'), target)
            unet_loss = 0.01*self.loss_func['criterion'](pred_probs.to('cpu'), dist_map) + gdl_loss
            unet_loss.backward()
            self.optimizers["unet"].step()
            training_loss['unet_loss'].append(unet_loss.item())

        training_loss = {key: sum(value) / len(value) for key, value in training_loss.items()}
        for loss_name, loss_value in training_loss.items():
            self.logger.add_scalar(f'train/{loss_name}', loss_value, epoch)
        self.save_checkpoints(os.path.join(log_dir, f'ckpt-epoch{epoch}.pth'))
        return training_loss


    def valid_epoch(self, valid_loader, epoch):
        self.models['gen_d2p'].eval()
        # TEST PER EPOCH
        valid_loss = []

        with torch.no_grad():
            for dic, _, target, _, _  in tqdm(valid_loader, desc='validating'):
                output = self.models['unet'](dic)
                valid_loss += self.loss_fn['l1_cyc'](output, target).item()
        
        # todo: merge_dics?
        valid_loss = merge_dicts(valid_loss)

        for loss_name, loss_value in valid_loss.items():
            self.logger.add_scalar(f'valid/{loss_name}', loss_value, epoch, new_style=True)
        return valid_loss

    def test_epoch(self, test_loader, epoch, device):
        self.models['unet'].eval()
        # TEST PER EPOCH
        #testing_loss = {'pix_acc':[], 'iou': [], 'boundary_iou':[], 'avgHausdorff':[], 'Hausdorff':[], 'dice': []}
        testing_loss = {'pix_acc':[], 'iou': [], 'boundary_iou':[]}
    
        with torch.no_grad():
            for dic, _, _, label in tqdm(test_loader, desc=f"testing epoch #{epoch}"):
                output = self.models['unet'](dic.to(device))
                y_eval = output.cpu().detach().numpy()
                y_eval = np.squeeze(y_eval)
                label = np.squeeze(label)
                y_eval = np.argmax(y_eval,axis=0)
                p_acc = pixel_accuracy(y_eval,label.detach().numpy())
                m_iu = mean_IU(y_eval,label.detach().numpy())
                b_iou = boundary_iou_after(label.detach().numpy(), y_eval)
                #avgHausdorff, Hausdorff, dice = computeQualityMeasures(y_eval, label.detach().numpy())
                testing_loss['pix_acc'].append(p_acc)
                testing_loss['iou'].append(m_iu)
                testing_loss['boundary_iou'].append(b_iou)
                #testing_loss['avgHausdorff'].append(avgHausdorff)
                #testing_loss['Hausdorff'].append(Hausdorff)
                #testing_loss['dice'].append(dice)
        testing_loss = {key: sum(value) / len(value) for key, value in testing_loss.items()}
        for loss_name, loss_value in testing_loss.items():
            self.logger.add_scalar(f'test/{loss_name}', loss_value, epoch, new_style=True)
        return testing_loss

    def _train(self, train_loader, test_loader, log_path, device):
        if self.retest:
            self.hparameters["metric_best"] = {}
            for metric in self.hparameters["metric"].keys():
                self.hparameters["metric"][metric] = []
            for epoch in range(1, 1 + self.hparameters["trained_epoch"]):
                print(f"retesting epoch {epoch} metric for trained model")
                self.load_checkpoints(
                    os.path.join(self.log_dir, f"ckpt-epoch{epoch}.pth"), device)
                te_losses = self.test_epoch(test_loader, epoch, device=device)

                for loss_type in te_losses.keys():
                    metric_value = float(te_losses[loss_type].mean())
                    if loss_type not in self.hparameters["metric"].keys():
                        self.hparameters["metric"].update({loss_type: [metric_value]})
                    else:
                        self.hparameters["metric"][loss_type].append(metric_value)
                    assert len(self.hparameters["metric"][loss_type]) == epoch
                    # self.hparameters[""]

        for epoch in range(1 + self.hparameters["trained_epoch"], 1 + self.hparameters["max_epoch"]):
            if self.hparameters["metric"]:
                early_stopping = self.pick_best_fn(self.hparameters["metric"][self.best_value_metric])
                if early_stopping + 1 < epoch - self.early_stopping_epoch:
                    print(f"early stopping at epoch {early_stopping + 1}")
                    break

            _ = self.train_epoch(train_loader, epoch, log_path, device)
            te_losses = self.test_epoch(test_loader, epoch, device)

            for loss_type in te_losses.keys():
                metric_value = float(te_losses[loss_type])
                if loss_type not in self.hparameters["metric"].keys():
                    self.hparameters["metric"].update({loss_type: [metric_value]})
                else:
                    self.hparameters["metric"][loss_type].append(metric_value)
                assert len(self.hparameters["metric"][loss_type]) == epoch
            self.hparameters["trained_epoch"] = epoch
            self.save_training_log(log_path, self.hparameters)

        self.hparameters["done"] = True
        self.save_training_log(log_path, self.hparameters)
        self.save_test_pred(test_loader)

    def train(self, train_loader, test_loader, device):
        # resume training and load models
        self._train(train_loader, test_loader, log_path=self.log_dir, device=device)

    def save_metric(self, log_path):
        best_value_idx = self.pick_best_fn(
            np.array(self.hparameters["metric"][self.best_value_metric])[:self.hparameters["max_epoch"]])
        for loss_type in self.hparameters["metric"].keys():
            self.hparameters["metric_best"][loss_type] = self.hparameters["metric"][loss_type][best_value_idx]
        self.hparameters["metric_best_epoch"] = int(best_value_idx + 1)
        # for key in self._hparameter_list:
        #     print(key, self.hparameters[key], type(self.hparameters[key]))
        self.logger.add_hparams(
            {key: self.hparameters[key] for key in self._hparameter_list if key not in ["metric", "metric_best"]},
            self.hparameters["metric_best"])
        self.save_training_log(log_path, self.hparameters)
    
    def save_test_pred(self, test_loader, device='cuda:4'):
        #best_epoch = self.hparameters["metric_best_epoch"]
        #self.load_checkpoints(os.path.join(self.log_dir, f"ckpt-epoch{best_epoch}.pth"), device)
        pred_dir = os.path.join(self.log_dir, 'pred')
        if not os.path.exists(pred_dir):
            os.mkdir(pred_dir)
        self.models['unet'].eval()
        i = 0
        testing_loss = {'pix_acc':[], 'iou': [], 'boundary_iou':[]}
        #pred=[]
        with torch.no_grad():
            for dic, _, _, label in tqdm(test_loader, desc=f"testing epoch"):

                output = self.models['unet'](dic.to(device))
                y_eval = output.cpu().detach().numpy()
                y_eval = np.squeeze(y_eval)
                label = np.squeeze(label)
                y_eval = np.argmax(y_eval, axis=0)
                #plt.imsave(os.path.join(pred_dir, 'state_{:03d}.png'.format(i)), y_eval)
                #pred.append(y_eval)
                p_acc = pixel_accuracy(y_eval, label.detach().numpy())
                m_iu = mean_IU(y_eval, label.detach().numpy())
                b_iou = boundary_iou(label.detach().numpy(), y_eval)

                testing_loss['pix_acc'].append(p_acc)
                testing_loss['iou'].append(m_iu)
                testing_loss['boundary_iou'].append(b_iou)
                i += 1
            #np.save(os.path.join(pred_dir, 'raw_dic_pred.npy'), pred)
            testing_loss = {key: sum(value) / len(value) for key, value in testing_loss.items()}
            print (testing_loss)