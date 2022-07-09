from cProfile import label
from sklearn.utils import shuffle
import torch
import torch.optim as optim
import os
from os import listdir
from skimage.io import imread
import pandas as pd
from Data_Loader import Dataset_folder
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
from Models import U_Net
from utils.losses import BoundaryLoss, GeneralizedDice, L1Loss, L2Loss
from Trainer import UnetTrainer
# transfer to trainer
# differ in iterate times
# combine gen and unet?

def run_main(args):
    
    # prepare tfb dir
    hparameter = {'seed': args.seed,
                  'max_epoch': args.n_epochs,
                  'lr': args.learning_rate,
                  'batch_size': args.train_batch_size,
                  }

    file_name = []
    for hp in hparameter.keys():
        if hp != "max_epoch":
            file_name.append(f"{hp}-{hparameter[hp]}")
    file_name = "-".join(file_name)
    log_dir = os.path.join(args.logdir, file_name)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    writer = SummaryWriter(log_dir=log_dir)
    writer.add_text('args', str(args))

    torch.manual_seed(args.seed)

    # models
    unet = U_Net(1, 3).to(args.device)
    
    # loss function
    dice_loss = GeneralizedDice(idc=[0,1,2])
    criterion = BoundaryLoss(idc=[2])

    # optimizer
    opt_unet = optim.Adam(unet.parameters(), lr=args.learning_rate)

   
    # MAX_STEP = int(1e10)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, MAX_STEP, eta_min=1e-5)
    dic_train = np.load(args.dic_train_img_dir)
    labels_train = np.load(args.dic_train_gt_dir)
    training_data = Dataset_folder(dic_train, labels_train)

    dic_test = np.load(args.dic_test_img_dir)
    labels_test = np.load(args.dic_test_gt_dir)
    testing_data = Dataset_folder(dic_test, labels_test)

    train_loader = torch.utils.data.DataLoader(training_data, batch_size=args.train_batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(testing_data, batch_size=args.test_batch_size)

    # num_train = len(training_data)
    # indices = list(range(num_train))
    # split = int(np.floor(args.valid_size * num_train))
    # train_idx, test_idx = indices[split:], indices[:split]
    # train_sampler = SubsetRandomSampler(train_idx)
    # test_sampler = SubsetRandomSampler(test_idx)
    # pin_memory = False
    # if torch.cuda.is_available():
    #     pin_memory = True
    # train_loader = torch.utils.data.DataLoader(training_data, batch_size=args.train_batch_size, sampler=train_sampler,
    #                                         num_workers=args.num_workers, pin_memory=pin_memory,)
    # # todo: add valid loader
    # test_loader = torch.utils.data.DataLoader(training_data, batch_size=args.test_batch_size, sampler=test_sampler,
    #                                         num_workers=args.num_workers, pin_memory=pin_memory,)

    
    trainer = UnetTrainer(models={"unet": unet},
                    loss_func={
                                'l1_cyc': torch.nn.L1Loss(),
                                'l1': L1Loss,
                                'l2': L2Loss,
                                'dice_loss': dice_loss,
                                'criterion': criterion},
                    optimizers={"unet": opt_unet},
                    batch_size=args.train_batch_size,
                    log_path=log_dir,
                    resume_training=args.resume_training,
                    hparameters=hparameter,
                    redo_metric=args.redo_metric,
                    retest=False,
                    early_stopping=15)

    need_to_train = trainer.start_training_log(True, args.device)

    if need_to_train:
        trainer.train(train_loader, test_loader, args.device)
        trainer.save_metric(log_dir)
    else:
        if args.redo_metric:
            print("recomputing metric")
            trainer.save_metric(log_dir)
        trainer.save_test_pred(test_loader)

    
# for argparse
def to_bool(value):
    valid = {'true': True, 't': True, '1': True,
             'false': False, 'f': False, '0': False,
             }

    if isinstance(value, bool):
        return value

    lower_value = value.lower()
    if lower_value in valid:
        return valid[lower_value]
    else:
        raise ValueError('invalid literal for boolean: "%s"' % value)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dic_train_gt_dir", type=str, default="/shared/home/v_zixin_tang/dataset/data256/dicTrainGT.npy")
    #parser.add_argument("--phase_train_img_dir", type=str, default="/shared/home/v_zixin_tang/dataset/NMuMg_crop/transferTrain.npy")
    parser.add_argument("--dic_train_img_dir", type=str, default="/shared/home/v_zixin_tang/dataset/data256/dicTrain.npy")
    parser.add_argument("--dic_test_gt_dir", type=str, default="/shared/home/v_zixin_tang/dataset/data256/dicTestGT.npy")
    parser.add_argument("--dic_test_img_dir", type=str, default="/shared/home/v_zixin_tang/dataset/data256/dicTest.npy")
    #parser.add_argument("--phase_test_img_dir", type=str, default="/shared/home/v_zixin_tang/dataset/NMuMg_crop/transferTest.npy")
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--test_batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--valid_size", type=float, default=0.1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=4)
    parser.add_argument("--redo_metric", type=to_bool, default=True)

    parser.add_argument("--save_img_per_epoch", type=to_bool, default=True)
    parser.add_argument("--log_interval", type=int, default=1)
    parser.add_argument("--device", type=str, default='cuda:1')
    parser.add_argument("--debug_mode", type=to_bool, default=False)
    parser.add_argument("--resume_training", type=to_bool, default=False)
    parser.add_argument("--train_test_split", type=float, default=0.9)  # Proportion train_set/full_data_set
    parser.add_argument("--logdir", type=str, default="tensorboard_runs")
    args = parser.parse_known_args()[0]
    run_main(args)