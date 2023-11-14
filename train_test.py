import os
import sys
from tqdm import tqdm
from tensorboardX import SummaryWriter
import shutil
import argparse
import logging
import time
import random
import numpy as np
import torch
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from networks.vnet import VNet
from networks.ResNet34 import Resnet34
from networks.unetr import UNETR
import yaml

# from networks.d_lka_former.d_lka_net_synapse import D_LKA_Net
# from networks.d_lka_former.transformerblock import TransformerBlock_3D_single_deform_LKA, TransformerBlock
from utils import ramps, losses
from dataloaders.la_heart import LAHeart, RandomCrop, ToTensor

# from main_model.models.dLKA import Model
from models.get_models import get_model
from utils_yousef import load_config, print_config
import os
from test_pancreas import test_calculate_metric

parser = argparse.ArgumentParser()
parser.add_argument(
    "--root_path",
    type=str,
    default="/cabinet/dataset/panceras",
    help="Name of Experiment",
)  # todo change dataset path
parser.add_argument(
    "--exp", type=str, default="pancreas", help="model_name"
)  # todo model name
parser.add_argument(
    "--max_iterations", type=int, default=6000, help="maximum epoch number to train"
)  # 6000
parser.add_argument("--batch_size", type=int, default=3, help="batch_size per gpu")
parser.add_argument(
    "--labeled_bs", type=int, default=1, help="labeled_batch_size per gpu"
)
parser.add_argument(
    "--base_lr", type=float, default=0.05, help="maximum epoch number to train"
)
parser.add_argument(
    "--deterministic", type=int, default=1, help="whether use deterministic training"
)
parser.add_argument("--seed", type=int, default=1337, help="random seed")
parser.add_argument("--gpu", type=str, default="0", help="GPU to use")
### costs
parser.add_argument("--ema_decay", type=float, default=0.999, help="ema_decay")
parser.add_argument(
    "--consistency_type", type=str, default="mse", help="consistency_type"
)
parser.add_argument("--consistency", type=float, default=0.1, help="consistency")
parser.add_argument(
    "--consistency_rampup", type=float, default=40.0, help="consistency_rampup"
)
parser.add_argument(
    "--config", type=str, default="./config.yaml", help="configuration file"
)

args = parser.parse_args()


def create_snapshot_directory(base_path, base_name):
    dir_number = 1
    dir_path = os.path.join(base_path, f"{base_name}{dir_number}/")

    # Check if the directory exists and increment the number if it does
    while os.path.exists(dir_path):
        dir_number += 1
        dir_path = os.path.join(base_path, f"{base_name}{dir_number}/")

    # Create the directory
    os.makedirs(dir_path)
    return dir_path


train_data_path = args.root_path
snapshot_path = "./model/" + args.exp + "/"

# os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
batch_size = args.batch_size * len(args.gpu.split(","))
max_iterations = args.max_iterations
base_lr = args.base_lr
labeled_bs = args.labeled_bs

config = load_config(args.config)
print_config(config)
train_data_path = config["dataset"]["train"]["params"]["base_dir"]
batch_size = config["data_loader"]["train"]["batch_size"]
max_iterations = config["max_iterations"]
base_lr = config["training"]["optimizer"]["params"]["lr"]
snapshot_path = config["snapshot_path"]
snapshot_path = create_snapshot_directory(snapshot_path, args.exp)
test_every_epochs = config["test_every_epochs"]

if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

num_classes = 2
patch_size = (96, 96, 96)  # 96x96x96 for Pancreas
T = 0.1
Good_student = 0  # 0: vnet 1:resnet


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def worker_init_fn(worker_id):
    random.seed(args.seed + worker_id)


def gateher_two_patch(vec):
    b, c, num = vec.shape
    cat_result = []
    for i in range(c - 1):
        temp_line = vec[:, i, :].unsqueeze(1)  # b 1 c
        star_index = i + 1
        rep_num = c - star_index
        repeat_line = temp_line.repeat(1, rep_num, 1)
        two_patch = vec[:, star_index:, :]
        temp_cat = torch.cat((repeat_line, two_patch), dim=2)
        cat_result.append(temp_cat)

    result = torch.cat(cat_result, dim=1)
    return result

def log_test_outputs(avg_metrics, logits, labels, writer, iter_num=0):
        writer.add_scalar("test/Dice_metric", avg_metrics[0], iter_num)
        writer.add_scalar("test/Jaccard_metric", avg_metrics[1], iter_num)
        writer.add_scalar("test/HD_metric", avg_metrics[2], iter_num)
        writer.add_scalar("test/ASD_metric", avg_metrics[3], iter_num)
        total_loss = []
        total_loss_seg = []
        total_loss_seg_dice = []    
        for logit,label in zip(logits,labels):
            ## calculate the supervised loss
            lka_loss_seg = F.cross_entropy(
                logit[:labeled_bs], label[:labeled_bs]
            )
            lka_outputs_soft = F.softmax(logit, dim=1)
            lka_loss_seg_dice = losses.dice_loss(
                lka_outputs_soft[:labeled_bs, 1, :, :, :], label[:labeled_bs] == 1
            )
            total_loss.append(lka_loss_seg + lka_loss_seg_dice)
            total_loss_seg.append(lka_loss_seg)
            total_loss_seg_dice.append(lka_loss_seg_dice)
        writer.add_scalar("test_loss/total_loss", np.mean(total_loss), iter_num)
        writer.add_scalar("test_loss/lka_loss_seg", np.mean(total_loss_seg), iter_num)
        writer.add_scalar("test_loss/lka_loss_seg_dice", np.mean(total_loss_seg_dice), iter_num)   

if __name__ == "__main__":
    ## make logger file
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    # if os.path.exists(snapshot_path + '/code'):
    #    shutil.rmtree(snapshot_path + '/code')
    # shutil.copytree('.', snapshot_path + '/code', shutil.ignore_patterns(['.git','__pycache__']))

    logging.basicConfig(
        filename=os.path.join(snapshot_path, "log.txt"),
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d] %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    # def create_model(name ='dlka_former'):
    #     # Network definition
    #     if name == 'dlka_former':
    #         net = D_LKA_Net(in_channels=1,
    #                        out_channels=num_classes,
    #                        img_size=[96, 96, 96],
    #                        patch_size=(2,2,2),
    #                        input_size=[48*48*48, 24*24*24,12*12*12,6*6*6],
    #                        trans_block=TransformerBlock_3D_single_deform_LKA,
    #                        do_ds=False)
    #         model = net.cuda()
    #     return model

    # model_d_lka_former = create_model(name='dlka_former')
    model_d_lka_former = get_model(config).cuda()
    # model_d_lka_former = Model(
    #         spatial_shapes= [96, 96, 96],
    #         in_channels= 1,
    #         out_channels=num_classes,
    #         # encoder params
    #         cnn_kernel_sizes= [7,5],
    #         cnn_features= [8,16],
    #         cnn_strides= [2,2],
    #         cnn_maxpools= [False, True],
    #         cnn_dropouts= 0.0,
    #         hyb_kernel_sizes= [5,5,5],
    #         hyb_features= [32,64,128],
    #         hyb_strides= [2,2,2],
    #         hyb_maxpools= [True, True, True],
    #         hyb_cnn_dropouts= 0.0,
    #         hyb_tf_proj_sizes= [32,64,64],
    #         hyb_tf_repeats= [2,2,1],
    #         hyb_tf_num_heads= [2,2,2],
    #         hyb_tf_dropouts= 0.15,

    #         # decoder params
    #         dec_hyb_tcv_kernel_sizes= [5,7,7],
    #         dec_cnn_tcv_kernel_sizes= [7,9],
    #     ).cuda()

    db_train = LAHeart(
        common_transform=transforms.Compose(
            [
                RandomCrop(patch_size),
            ]
        ),
        sp_transform=transforms.Compose(
            [
                ToTensor(),
            ]
        ),
        **config["dataset"]["train"]["params"],
    )

    trainloader = DataLoader(
        db_train, worker_init_fn=worker_init_fn, **config["data_loader"]["train"]
    )
    optimizer_cls = getattr(torch.optim, config["training"]["optimizer"]["name"])
    d_lka_former_optimizer = optimizer_cls(
        model_d_lka_former.parameters(), **config["training"]["optimizer"]["params"]
    )

    writer = SummaryWriter(snapshot_path + "/log")
    logging.info("{} itertations per epoch".format(len(trainloader)))
    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    lr_ = base_lr


    with open(
        os.path.join(train_data_path, "Pancreas", "Flods", config["dataset"]["test"]["params"]['test_flod']), "r"
    ) as f:  # todo change test flod
        image_list = f.readlines()
    image_list = [
        os.path.join(train_data_path, item.replace("\n", "")) for item in image_list
    ]
    
    with open(os.path.join(snapshot_path, "hpram.yaml"), "w") as yaml_file:
        yaml.dump(config, yaml_file)

    for epoch_num in tqdm(range(max_epoch), ncols=70):
        time1 = time.time()
        model_d_lka_former.train()
        for i_batch, sampled_batch in enumerate(trainloader):
            time2 = time.time()
            #             print('epoch:{}, i_batch:{}'.format(epoch_num,i_batch))
            volume_batch1, volume_label1 = (
                sampled_batch[0]["image"],
                sampled_batch[0]["label"],
            )
            volume_batch2, volume_label2 = (
                sampled_batch[1]["image"],
                sampled_batch[1]["label"],
            )
            # Transfer to GPU
            lka_input, lka_label = volume_batch1.cuda(), volume_label1.cuda()

            # Network forward
            lka_outputs = model_d_lka_former(lka_input)

            ## calculate the supervised loss
            lka_loss_seg = F.cross_entropy(
                lka_outputs[:labeled_bs], lka_label[:labeled_bs]
            )
            lka_outputs_soft = F.softmax(lka_outputs, dim=1)
            lka_loss_seg_dice = losses.dice_loss(
                lka_outputs_soft[:labeled_bs, 1, :, :, :], lka_label[:labeled_bs] == 1
            )

            loss_total = lka_loss_seg + lka_loss_seg_dice
            # Network backward
            d_lka_former_optimizer.zero_grad()
            loss_total.backward()
            d_lka_former_optimizer.step()

            writer.add_scalar("lr", lr_, iter_num)
            writer.add_scalar("train_loss/total_loss", loss_total, iter_num)
            writer.add_scalar("train_loss/lka_loss_seg", lka_loss_seg, iter_num)
            writer.add_scalar("train_loss/lka_loss_seg_dice", lka_loss_seg_dice, iter_num)

            if iter_num % 50 == 0 and iter_num != 0:
                logging.info(
                    "iteration: %d Total loss : %f CE loss : %f Dice loss : %f"
                    % (
                        iter_num,
                        loss_total.item(),
                        lka_loss_seg.item(),
                        lka_loss_seg_dice.item(),
                    )
                )

            ## change lr
            if iter_num % 2500 == 0 and iter_num != 0:
                lr_ = lr_ * 0.1
                for param_group in d_lka_former_optimizer.param_groups:
                    param_group["lr"] = lr_

            if iter_num >= max_iterations:
                break
            time1 = time.time()

            iter_num = iter_num + 1
            if iter_num >= max_iterations:
                break

        if (epoch_num <= 100 and epoch_num % 50 == 0 and epoch_num != 0) or (
            epoch_num > 100 and epoch_num % test_every_epochs == 0
        ):
            save_mode_path_lka_net = os.path.join(
                snapshot_path,
                f"epoch_{str(epoch_num)}",
                f"d_lka_former_iter_{str(epoch_num)}.pth",
            )
            os.makedirs(
                os.path.join(snapshot_path, f"epoch_{str(epoch_num)}"), exist_ok=True
            )
            torch.save(model_d_lka_former.state_dict(), save_mode_path_lka_net)
            logging.info("save model to {}".format(save_mode_path_lka_net))
            # testing
            # the model automatically goes to the eval mode
            test_save_path = os.path.join(
                snapshot_path, f"epoch_{str(epoch_num)}", "prediction"
            )
            os.makedirs(test_save_path, exist_ok=True)
            avg_metrics, logits, labels = test_calculate_metric(
                epoch_num, model_d_lka_former, snapshot_path, test_save_path, image_list
            )
            log_test_outputs(avg_metrics, logits, labels, writer, iter_num=iter_num)

        if iter_num >= max_iterations:
            break
        ###

    save_mode_path_lka_net = os.path.join(
        snapshot_path,
        f"epoch_{str(max_iterations)}",
        f"d_lka_former_iter_{str(max_iterations)}.pth",
    )
    torch.save(model_d_lka_former.state_dict(), save_mode_path_lka_net)
    logging.info("save model to {}".format(save_mode_path_lka_net))
    test_save_path = os.path.join(
        snapshot_path, f"epoch_{str(max_iterations)}", "prediction"
    )
    os.makedirs(test_save_path, exist_ok=True)
    metric, logits, labels = test_calculate_metric(
        max_iterations, model_d_lka_former, snapshot_path, test_save_path, image_list
    )
    log_test_outputs(metric, logits, labels, writer, iter_num=max_iterations)
    print("iter:", max_iterations)
    print(metric)

    writer.close()
