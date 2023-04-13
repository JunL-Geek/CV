import datetime

from utils.utils import get_classes, show_config
from frcnn import FasterRCNN
from frcnn_training import weights_init, get_lr_scheduler, FasterRCNNTrainer, set_optimizer_lr
import os
import torch
import numpy as np
import torch.backends.cudnn as cudnn
from torch import optim
from utils.dataloader import FRCNNDataset, frcnn_dataset_collate
from torch.utils.data import DataLoader
from utils.callbacks import EvalCallback, LossHistory
from utils.utils_fit import fit_one_epoch


if __name__ == "__main__":

    Cuda = True
    train_gpu = [0, ]
    fp16 = False
    classes_path = 'model_data/voc_classes.txt'
    model_path = 'model_data/voc_weights_resnet.pth'
    input_shape = [600, 600]
    backbone = 'resnet50'
    pretrained = False
    anchor_size = [8, 16, 32]
    Init_Epoch = 0
    Freeze_Epoch = 50
    Freeze_batch_size = 4
    Unfreeze_Epoch = 100
    Unfreeze_batch_size = 2
    Freeze_Train = True
    Init_lr = 1e-4
    Min_lr = Init_lr * 0.01
    optimizer_type = 'adam'
    momentum = 0.9
    weight_decay = 0
    lr_decay_type = 'cos'
    save_period = 5
    save_dir = 'logs'
    eval_flag = True
    eval_period = 5
    num_workers = 1

    train_annotion_path = '2007_train.txt'
    val_annotion_path = '2007_val.txt'

    class_names, num_classes = get_classes(classes_path)

    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in train_gpu)
    ngpus_per_node = len(train_gpu)
    print('Number of devices: {}'.format(ngpus_per_node))

    model = FasterRCNN(num_classes, anchor_scales=anchor_size, backbone=backbone, pretrained=pretrained)
    if not pretrained:
        weights_init(model)
    if model_path != '':
        print('load weights {}'.format(model_path))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_dict = model.state_dict()
        predicted_dict = torch.load(model_path, map_location = device)
        temp_dict = {}
        load_key, no_load_key = [], []
        for k, v in predicted_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                load_key.append(k)
                temp_dict[k] = v
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)

        # 显示成功加载的模型参数
        print('\n Successful Load Key:', str(load_key)[:500], '......\nSuccessful Load Key Num', len(load_key))
        print('\nFail to Load Key:', str(no_load_key)[:500], '......\nFail to Load Key Num', len(no_load_key))
        print("head 部分参数载入失败为正常, Backbone 参数载入失败为异常")

    time_str = datetime.datetime.strftime(datetime.datetime.now(), '%y_%m_%d_%H_%M_%S')
    log_dir = os.path.join(save_dir, "loss_" + str(time_str))
    loss_history = LossHistory(log_dir, model, input_shape=input_shape)

    if fp16:
        from torch.cuda.amp import GradScaler as GradScaler
        scaler = GradScaler()
    else:
        scaler = None

    model_train = model.train()
    if Cuda:
        model_train = torch.nn.DataParallel(model_train)
        cudnn.benchmark = True
        model_train = model_train.cuda()


    with open(train_annotion_path, encoding='utf-8') as f:
        train_lines = f.readlines()
    with open(val_annotion_path, encoding='utf-8') as f:
        val_lines = f.readlines()
    num_train = len(train_lines)
    num_val = len(val_lines)

    show_config(
        classes_path = classes_path, model_path = model_path, input_shape = input_shape, \
        Init_Epoch = Init_Epoch, Freeze_Epoch = Freeze_Epoch, Unfreeze_Epoch = Unfreeze_Epoch, Freeze_batch_size = Freeze_batch_size, Unfreeze_batch_size=Unfreeze_batch_size, Freeze_Train = Freeze_Train,\
        Init_lr = Init_lr, Min_lr = Min_lr, optimizer_type = optimizer_type, momentum = momentum, lr_decay_type = lr_decay_type, \
        save_period = save_period, save_dir = save_dir, num_workers = num_workers, num_train = num_train, num_val = num_val
    )

    wanted_step = 5e4 if optimizer_type == 'sgd' else 1.5e4
    total_step = num_train // Unfreeze_batch_size * Unfreeze_Epoch
    if total_step <= wanted_step:
        if num_train // Unfreeze_batch_size == 0:
            raise ValueError("数据集过小， 无法进行训练， 请扩充数据集")
        wanted_epoch = wanted_step // (num_train // Unfreeze_batch_size) + 1

    if True:
        Unfreeze_flag = False

        if Freeze_Train:
            for param in model.extractor.parameters():
                param.requires_grad = False

        model.freeze_bn()

        batch_size = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size

        nbs = 16
        lr_limit_max = 1e-4 if optimizer_type == 'adam' else 5e-2
        lr_limit_min = 1e-4 if optimizer_type == 'adam' else 5e-4
        Init_lr_fit = min(max(batch_size/nbs * Init_lr, lr_limit_min), lr_limit_max)
        Min_lr_fit = min(max(batch_size/nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

        optimizer = {
            'adam':optim.Adam(model.parameters(), Init_lr_fit, betas = (momentum, 0.999), weight_decay = weight_decay),
            'sgd' :optim.SGD(model.parameters(), Init_lr_fit, momentum=momentum, nesterov=True, weight_decay=weight_decay)
        }[optimizer_type]

        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, Unfreeze_Epoch)

        epoch_step = num_train // batch_size
        epoch_step_val = num_val // batch_size

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("数据集过小， 无法进行训练， 请扩充数据集")

        train_dataset = FRCNNDataset(train_lines, input_shape, train=True)
        val_dataset = FRCNNDataset(val_lines, input_shape, train=False)

        gen = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                         drop_last=True, collate_fn=frcnn_dataset_collate)
        gen_val = DataLoader(val_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                             drop_last=True, collate_fn=frcnn_dataset_collate)

        train_util = FasterRCNNTrainer(model_train, optimizer)

        #记录eval的map曲线
        eval_callback = EvalCallback(model_train, input_shape, class_names, num_classes, val_lines, log_dir, Cuda, \
                                     eval_flag=eval_flag, period=eval_period)


        #开始模型训练
        for epoch in range(Init_Epoch, Unfreeze_Epoch):
            if epoch >= Freeze_Epoch and not Unfreeze_flag and Freeze_Train:
                batch_size = Unfreeze_batch_size
                # -------------------------------------------------------------------#
                #   判断当前batch_size，自适应调整学习率
                # -------------------------------------------------------------------#
                nbs = 16
                lr_limit_max = 1e-4 if optimizer_type == 'adam' else 5e-2
                lr_limit_min = 1e-4 if optimizer_type == 'adam' else 5e-4
                Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
                Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
                # ---------------------------------------#
                #   获得学习率下降的公式
                # ---------------------------------------#
                lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, Unfreeze_Epoch)

                for param in model.extractor.parameters():
                    param.requires_grad = True
                # ------------------------------------#
                #   冻结bn层
                # ------------------------------------#
                model.freeze_bn()

                epoch_step = num_train // batch_size
                epoch_step_val = num_val // batch_size

                if epoch_step == 0 or epoch_step_val == 0:
                    raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")

                gen = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers,
                                 pin_memory=True,
                                 drop_last=True, collate_fn=frcnn_dataset_collate)
                gen_val = DataLoader(val_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers,
                                     pin_memory=True,
                                     drop_last=True, collate_fn=frcnn_dataset_collate)

                UnFreeze_flag = True

            set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

            fit_one_epoch(model, train_util, loss_history, eval_callback, optimizer, epoch, epoch_step, epoch_step_val,
                          gen, gen_val, Unfreeze_Epoch, Cuda, fp16, scaler, save_period, save_dir)

        loss_history.writer.close()