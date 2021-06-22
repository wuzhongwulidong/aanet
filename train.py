import torch
from torch.cuda import synchronize
from torch.utils.data import DataLoader
from torch import distributed

import argparse
import numpy as np
import os

import nets
import dataloader
from dataloader import transforms
from dataloader.dataloader import getDataLoader
from utils import utils
import model
from utils.utilsForMatlab import getLossRecord, save_loss_for_matlab

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

parser = argparse.ArgumentParser()

parser.add_argument('--mode', default='val', type=str,
                    help='Validation mode on small subset or test mode on full test data')
# 用于选择数据集：0: debug; 1: overFit; 2: Train
parser.add_argument('--debug_overFit_train', default=1, type=int, help='For code debug only!')

# Training data
parser.add_argument('--data_dir', default='data/SceneFlow', type=str, help='Training dataset')
parser.add_argument('--dataset_name', default='SceneFlow', type=str, help='Dataset name')

parser.add_argument('--batch_size', default=64, type=int, help='Batch size for training')
parser.add_argument('--accumulation_steps', default=1, type=int, help='Batch size for training')
parser.add_argument('--val_batch_size', default=64, type=int, help='Batch size for validation')
parser.add_argument('--num_workers', default=8, type=int, help='Number of workers for data loading')
parser.add_argument('--img_height', default=288, type=int, help='Image height for training')
parser.add_argument('--img_width', default=512, type=int, help='Image width for training')

# For KITTI, using 384x1248 for validation
parser.add_argument('--val_img_height', default=576, type=int, help='Image height for validation')
parser.add_argument('--val_img_width', default=960, type=int, help='Image width for validation')

# Model
parser.add_argument('--seed', default=326, type=int, help='Random seed for reproducibility')
parser.add_argument('--checkpoint_dir', default=None, type=str, required=True,
                    help='Directory to save model checkpoints and logs')
parser.add_argument('--learning_rate', default=1e-3, type=float, help='Learning rate')
parser.add_argument('--weight_decay', default=1e-4, type=float, help='Weight decay for optimizer')
parser.add_argument('--max_disp', default=192, type=int, help='Max disparity')
parser.add_argument('--max_epoch', default=64, type=int, help='Maximum epoch number for training')
parser.add_argument('--resume', action='store_true', help='Resume training from latest checkpoint')

# AANet
parser.add_argument('--useFeatureAtt', default=1, type=int, help='Whether to use Feature Attention: 1:use; 0:dont use')
parser.add_argument('--feature_type', default='aanet', type=str, help='Type of feature extractor')
parser.add_argument('--no_feature_mdconv', action='store_true', help='Whether to use mdconv for feature extraction')
parser.add_argument('--feature_pyramid', action='store_true', help='Use pyramid feature')
parser.add_argument('--feature_pyramid_network', action='store_true', help='Use FPN')
parser.add_argument('--feature_similarity', default='correlation', type=str,
                    help='Similarity measure for matching cost')
parser.add_argument('--num_downsample', default=2, type=int, help='Number of downsample layer for feature extraction')
parser.add_argument('--aggregation_type', default='adaptive', type=str, help='Type of cost aggregation')
parser.add_argument('--num_scales', default=3, type=int, help='Number of stages when using parallel aggregation')
parser.add_argument('--num_fusions', default=6, type=int, help='Number of multi-scale fusions when using parallel'
                                                               'aggragetion')
parser.add_argument('--num_stage_blocks', default=1, type=int, help='Number of deform blocks for ISA')
parser.add_argument('--num_deform_blocks', default=3, type=int, help='Number of DeformBlocks for aggregation')
parser.add_argument('--no_intermediate_supervision', action='store_true',
                    help='Whether to add intermediate supervision')
parser.add_argument('--deformable_groups', default=2, type=int, help='Number of deformable groups')
parser.add_argument('--mdconv_dilation', default=2, type=int, help='Dilation rate for deformable conv')
parser.add_argument('--refinement_type', default='stereodrnet', help='Type of refinement module')

parser.add_argument('--pretrained_aanet', default=None, type=str, help='Pretrained network')
parser.add_argument('--freeze_bn', action='store_true', help='Switch BN to eval mode to fix running statistics')

# Learning rate
parser.add_argument('--lr_decay_gamma', default=0.5, type=float, help='Decay gamma')
parser.add_argument('--lr_scheduler_type', default='MultiStepLR', help='Type of learning rate scheduler')
parser.add_argument('--milestones', default=None, type=str, help='Milestones for MultiStepLR')

# Loss
parser.add_argument('--highest_loss_only', action='store_true', help='Only use loss on highest scale for finetuning')
parser.add_argument('--load_pseudo_gt', action='store_true', help='Load pseudo gt for supervision')

# Log
parser.add_argument('--print_freq', default=50, type=int, help='Print frequency to screen (iterations)')
parser.add_argument('--summary_freq', default=100, type=int, help='Summary frequency to tensorboard (iterations)')
parser.add_argument('--no_build_summary', action='store_true', help='Dont save sammary when training to save space')
parser.add_argument('--save_ckpt_freq', default=5, type=int, help='Save checkpoint frequency (epochs)')

parser.add_argument('--evaluate_only', action='store_true', help='Evaluate pretrained models')
parser.add_argument('--no_validate', action='store_true', help='No validation')
parser.add_argument('--strict', action='store_true', help='Strict mode when loading checkpoints')
parser.add_argument('--val_metric', default='epe', help='Validation metric to select best model')

#  尝试分布式训练:
parser.add_argument("--local_rank", type=int)  # 必须有这一句，但是local_rank是torch.distributed.launch自动分配和传入的。
parser.add_argument("--distributed", action='store_true', help="use DistributedDataParallel")

args = parser.parse_args()
utils.check_path(args.checkpoint_dir)
logger = utils.get_logger(os.path.join(args.checkpoint_dir, "trainLog.txt"))

# 调整打印频率
if args.debug_overFit_train in [0, 2]:
    args.print_freq = 50
    args.summary_freq = 100
elif args.debug_overFit_train in [1]:
    args.print_freq = 10
    args.summary_freq = 50

if args.dataset_name == 'KITTI2015':
    args.print_freq = 20

if args.distributed:
    #  尝试分布式训练
    # local_rank = torch.distributed.get_rank()
    # local_rank表示本台机器上的进程序号,是由torch.distributed.launch自动分配和传入的。
    local_rank = args.local_rank
    # 根据local_rank来设定当前使用哪块GPU
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    # 初始化DDP，使用默认backend(nccl)就行
    torch.distributed.init_process_group(backend="nccl")
    print("args.local_rank={}".format(args.local_rank))
else:
    device = torch.device("cuda")

# 尝试分布式训练
local_master = True if not args.distributed else args.local_rank == 0
utils.save_args(args) if local_master else None

# 打印所用的参数
if local_master:
    logger.info('[Info] used parameters: {}'.format(vars(args)))

torch.backends.cudnn.benchmark = True  # https://blog.csdn.net/byron123456sfsfsfa/article/details/96003317

utils.check_path(args.checkpoint_dir)
utils.save_args(args) if local_master else None

filename = 'command_test.txt' if args.mode == 'test' else 'command_train.txt'
utils.save_command(args.checkpoint_dir, filename) if local_master else None


def main():
    # For reproducibility
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_loader, val_loader = getDataLoader(args, logger)

    # Network
    aanet = nets.AANet(args.max_disp,
                       num_downsample=args.num_downsample,
                       feature_type=args.feature_type,
                       no_feature_mdconv=args.no_feature_mdconv,
                       feature_pyramid=args.feature_pyramid,
                       feature_pyramid_network=args.feature_pyramid_network,
                       feature_similarity=args.feature_similarity,
                       aggregation_type=args.aggregation_type,
                       useFeatureAtt=args.useFeatureAtt,
                       num_scales=args.num_scales,
                       num_fusions=args.num_fusions,
                       num_stage_blocks=args.num_stage_blocks,
                       num_deform_blocks=args.num_deform_blocks,
                       no_intermediate_supervision=args.no_intermediate_supervision,
                       refinement_type=args.refinement_type,
                       mdconv_dilation=args.mdconv_dilation,
                       deformable_groups=args.deformable_groups).to(device)

    # logger.info('%s' % aanet) if local_master else None
    if local_master:
        structure_of_net = os.path.join(args.checkpoint_dir, 'structure_of_net.txt')
        with open(structure_of_net, 'w') as f:
            f.write('%s' % aanet)

    if args.pretrained_aanet is not None:
        logger.info('=> Loading pretrained AANet: %s' % args.pretrained_aanet)
        # Enable training from a partially pretrained model
        utils.load_pretrained_net(aanet, args.pretrained_aanet, no_strict=(not args.strict))

    aanet.to(device)
    logger.info('=> Use %d GPUs' % torch.cuda.device_count()) if local_master else None
    # if torch.cuda.device_count() > 1:
    if args.distributed:
        # aanet = torch.nn.DataParallel(aanet)
        #  尝试分布式训练
        aanet = torch.nn.SyncBatchNorm.convert_sync_batchnorm(aanet)
        aanet = torch.nn.parallel.DistributedDataParallel(aanet, device_ids=[local_rank],
                                                          output_device=local_rank)
        synchronize()

    # Save parameters
    num_params = utils.count_parameters(aanet)
    logger.info('=> Number of trainable parameters: %d' % num_params)
    save_name = '%d_parameters' % num_params
    open(os.path.join(args.checkpoint_dir, save_name), 'a').close() if local_master else None # 这是个空文件，只是通过其文件名称指示模型有多少个需要训练的参数

    # Optimizer
    # Learning rate for offset learning is set 0.1 times those of existing layers
    specific_params = list(filter(utils.filter_specific_params,
                                  aanet.named_parameters()))
    base_params = list(filter(utils.filter_base_params,
                              aanet.named_parameters()))

    specific_params = [kv[1] for kv in specific_params]  # kv is a tuple (key, value)
    base_params = [kv[1] for kv in base_params]

    specific_lr = args.learning_rate * 0.1
    params_group = [
        {'params': base_params, 'lr': args.learning_rate},
        {'params': specific_params, 'lr': specific_lr},
    ]

    optimizer = torch.optim.Adam(params_group, weight_decay=args.weight_decay)

    # Resume training
    if args.resume:
        # 1. resume AANet
        start_epoch, start_iter, best_epe, best_epoch = utils.resume_latest_ckpt(
            args.checkpoint_dir, aanet, 'aanet')
        # 2. resume Optimizer
        utils.resume_latest_ckpt(args.checkpoint_dir, optimizer, 'optimizer')
    else:
        start_epoch = 0
        start_iter = 0
        best_epe = None
        best_epoch = None

    # LR scheduler
    if args.lr_scheduler_type is not None:
        last_epoch = start_epoch if args.resume else start_epoch - 1
        if args.lr_scheduler_type == 'MultiStepLR':
            milestones = [int(step) for step in args.milestones.split(',')]
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                                milestones=milestones,
                                                                gamma=args.lr_decay_gamma,
                                                                last_epoch=last_epoch)  # 最后这个last_epoch参数很重要：如果是resume的话，则会自动调整学习率适去应last_epoch。
            logger.info('=>lr_scheduler.get_lr():{}'.format(lr_scheduler.state_dict()))
        else:
            raise NotImplementedError
    # model.Model(object)对AANet做了进一步封装。
    train_model = model.Model(args, logger, optimizer, aanet, device, start_iter, start_epoch,
                              best_epe=best_epe, best_epoch=best_epoch)

    logger.info('=> Start training...')

    trainLoss_dict, trainLossKey, valLoss_dict, valLossKey = getLossRecord(netName="AANet")

    if args.evaluate_only:
        assert args.val_batch_size == 1
        train_model.validate(val_loader, local_master, valLoss_dict, valLossKey)  # test模式。应该设置--evaluate_only，且--mode为“test”。
        # 保存Loss用于分析
        save_loss_for_matlab(trainLoss_dict, valLoss_dict)
    else:
        for epoch in range(start_epoch, args.max_epoch):  # 训练主循环（Epochs）！！！
            if not args.evaluate_only:
                # ensure distribute worker sample different data,
                # set different random seed by passing epoch to sampler
                if args.distributed:
                    train_loader.sampler.set_epoch(epoch)
                    logger.info('train_loader.sampler.set_epoch({})'.format(epoch))
                train_model.train(train_loader, local_master, trainLoss_dict, trainLossKey)

            kittiMixVal = ("KITTI_mix" != args.dataset_name and "KITTI2015" != args.dataset_name) \
                or ((epoch + 1) % 5 == 0 or epoch+1 == 1)
            if not args.no_validate and args.debug_overFit_train != 1 and kittiMixVal:
                train_model.validate(val_loader, local_master, valLoss_dict, valLossKey)  # 训练模式下：边训练边验证。
            if args.lr_scheduler_type is not None:
                lr_scheduler.step()  # 调整Learning Rate

            # 保存Loss用于分析。每个epoch结束后，都保存一次，覆盖之前的保存。避免必须训练完成才保存的弊端。
            save_loss_for_matlab(trainLoss_dict, valLoss_dict)

        logger.info('=> End training\n\n')


if __name__ == '__main__':
    main()
