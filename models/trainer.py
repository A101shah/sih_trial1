import numpy as np
import matplotlib.pyplot as plt
import os

import utils
from models.networks import *

import torch
import torch.optim as optim

from misc.metric_tool import ConfuseMatrixMeter
from models.losses import cross_entropy
import models.losses as losses
from models.losses import (
    get_alpha,
    softmax_helper,
    FocalLoss,
    mIoULoss,
    mmIoULoss
)

from misc.logger_tool import Logger, Timer

from utils import de_norm

from tqdm import tqdm


class CDTrainer():

    def __init__(self, args, dataloaders):

        self.args = args
        self.dataloaders = dataloaders

        self.n_class = args.n_class

        # --------------------------------------------------
        # Define network
        # --------------------------------------------------

        self.net_G = define_G(
            args=args,
            gpu_ids=args.gpu_ids
        )

        self.device = torch.device(
            "cuda:%s" % args.gpu_ids[0]
            if torch.cuda.is_available() and len(args.gpu_ids) > 0
            else "cpu"
        )

        print("Device:", self.device)

        # --------------------------------------------------
        # Learning rate
        # --------------------------------------------------

        self.lr = args.lr

        # --------------------------------------------------
        # Optimizer
        # --------------------------------------------------

        if args.optimizer == "sgd":

            self.optimizer_G = optim.SGD(
                self.net_G.parameters(),
                lr=self.lr,
                momentum=0.9,
                weight_decay=5e-4
            )

        elif args.optimizer == "adam":

            self.optimizer_G = optim.Adam(
                self.net_G.parameters(),
                lr=self.lr,
                weight_decay=0
            )

        elif args.optimizer == "adamw":

            self.optimizer_G = optim.AdamW(
                self.net_G.parameters(),
                lr=self.lr,
                betas=(0.9, 0.999),
                weight_decay=0.01
            )

        else:
            raise ValueError(
                "Unknown optimizer: " + args.optimizer
            )

        # --------------------------------------------------
        # Learning rate scheduler
        # --------------------------------------------------

        self.exp_lr_scheduler_G = get_scheduler(
            self.optimizer_G,
            args
        )

        # --------------------------------------------------
        # Metrics
        # --------------------------------------------------

        self.running_metric = ConfuseMatrixMeter(
            n_class=2
        )

        # --------------------------------------------------
        # Logger
        # --------------------------------------------------

        logger_path = os.path.join(
            args.checkpoint_dir,
            'log.txt'
        )

        self.logger = Logger(
            logger_path
        )

        self.logger.write_dict_str(
            args.__dict__
        )

        # --------------------------------------------------
        # Timer
        # --------------------------------------------------

        self.timer = Timer()

        self.batch_size = args.batch_size

        # --------------------------------------------------
        # Training variables
        # --------------------------------------------------

        self.epoch_acc = 0

        self.best_val_acc = 0.0

        self.best_epoch_id = 0

        self.epoch_to_start = 0

        self.max_num_epochs = args.max_epochs

        self.global_step = 0

        self.steps_per_epoch = len(
            dataloaders['train']
        )

        self.total_steps = (
            self.max_num_epochs -
            self.epoch_to_start
        ) * self.steps_per_epoch

        self.G_pred = None

        self.pred_vis = None

        self.batch = None

        self.G_loss = None

        self.is_training = False

        self.batch_id = 0

        self.epoch_id = 0

        self.checkpoint_dir = args.checkpoint_dir

        self.vis_dir = args.vis_dir

        self.shuffle_AB = args.shuffle_AB

        # --------------------------------------------------
        # Loss
        # --------------------------------------------------

        self.multi_scale_train = args.multi_scale_train

        self.multi_scale_infer = args.multi_scale_infer

        self.weights = tuple(
            args.multi_pred_weights
        )

        if args.loss == 'ce':

            self._pxl_loss = cross_entropy

        elif args.loss == 'bce':

            self._pxl_loss = losses.binary_ce

        elif args.loss == 'fl':

            print(
                '\nCalculating alpha in Focal-Loss (FL) ...'
            )

            alpha = get_alpha(
                dataloaders['train']
            )

            print(
                f"alpha-0 (no-change)={alpha[0]}, "
                f"alpha-1 (change)={alpha[1]}"
            )

            self._pxl_loss = FocalLoss(
                apply_nonlin=softmax_helper,
                alpha=alpha,
                gamma=2,
                smooth=1e-5
            )

        elif args.loss == "miou":

            print(
                '\nCalculating Class occurrences in training set...'
            )

            alpha = np.asarray(
                get_alpha(
                    dataloaders['train']
                )
            )

            alpha = alpha / np.sum(alpha)

            weights = (
                1 -
                torch.from_numpy(alpha).cuda()
            )

            print(
                f"Weights = {weights}"
            )

            self._pxl_loss = mIoULoss(
                weight=weights,
                size_average=True,
                n_classes=args.n_class
            ).cuda()

        elif args.loss == "mmiou":

            self._pxl_loss = mmIoULoss(
                n_classes=args.n_class
            ).cuda()

        else:

            raise NotImplementedError(
                args.loss
            )

        # --------------------------------------------------
        # Accuracy curves
        # --------------------------------------------------

        self.VAL_ACC = np.array(
            [],
            np.float32
        )

        if os.path.exists(
            os.path.join(
                self.checkpoint_dir,
                'val_acc.npy'
            )
        ):

            self.VAL_ACC = np.load(
                os.path.join(
                    self.checkpoint_dir,
                    'val_acc.npy'
                )
            )

        self.TRAIN_ACC = np.array(
            [],
            np.float32
        )

        if os.path.exists(
            os.path.join(
                self.checkpoint_dir,
                'train_acc.npy'
            )
        ):

            self.TRAIN_ACC = np.load(
                os.path.join(
                    self.checkpoint_dir,
                    'train_acc.npy'
                )
            )

        # --------------------------------------------------
        # Create directories
        # --------------------------------------------------

        if not os.path.exists(
            self.checkpoint_dir
        ):

            os.makedirs(
                self.checkpoint_dir
            )

        if not os.path.exists(
            self.vis_dir
        ):

            os.makedirs(
                self.vis_dir
            )


    # ======================================================
    # CHECKPOINT LOADING
    # ======================================================

    def _load_checkpoint(
        self,
        ckpt_name='last_ckpt.pt'
    ):

        print("\n")

        last_checkpoint = os.path.join(
            self.checkpoint_dir,
            ckpt_name
        )

        # --------------------------------------------------
        # Resume training if last checkpoint exists
        # --------------------------------------------------

        if os.path.exists(
            last_checkpoint
        ):

            self.logger.write(
                'Loading last checkpoint...\n'
            )

            checkpoint = torch.load(
                last_checkpoint,
                map_location=self.device,
                weights_only=False
            )

            self.net_G.load_state_dict(
                checkpoint[
                    'model_G_state_dict'
                ]
            )

            self.optimizer_G.load_state_dict(
                checkpoint[
                    'optimizer_G_state_dict'
                ]
            )

            self.exp_lr_scheduler_G.load_state_dict(
                checkpoint[
                    'exp_lr_scheduler_G_state_dict'
                ]
            )

            self.net_G.to(
                self.device
            )

            self.epoch_to_start = (
                checkpoint['epoch_id'] + 1
            )

            self.best_val_acc = (
                checkpoint['best_val_acc']
            )

            self.best_epoch_id = (
                checkpoint['best_epoch_id']
            )

            self.total_steps = (
                self.max_num_epochs -
                self.epoch_to_start
            ) * self.steps_per_epoch

            self.logger.write(
                'Epoch_to_start = %d, '
                'Historical_best_acc = %.4f '
                '(at epoch %d)\n'
                % (
                    self.epoch_to_start,
                    self.best_val_acc,
                    self.best_epoch_id
                )
            )

            self.logger.write('\n')

        # --------------------------------------------------
        # Initialize from pretrained model
        # --------------------------------------------------

        elif self.args.pretrain is not None:

            print(
                "Initializing pretrained weights from:"
            )

            print(
                self.args.pretrain
            )

            checkpoint = torch.load(
                self.args.pretrain,
                map_location=self.device,
                weights_only=False
            )

            # --------------------------------------------------
            # IMPORTANT:
            # ChangeFormer best_ckpt.pt contains a dictionary.
            # Extract model_G_state_dict.
            # --------------------------------------------------

            if isinstance(
                checkpoint,
                dict
            ) and 'model_G_state_dict' in checkpoint:

                print(
                    "Detected ChangeFormer training checkpoint."
                )

                checkpoint = checkpoint[
                    'model_G_state_dict'
                ]

            else:

                print(
                    "Detected raw model state dictionary."
                )

            # --------------------------------------------------
            # Load model weights
            # --------------------------------------------------

            missing_keys, unexpected_keys = \
                self.net_G.load_state_dict(
                    checkpoint,
                    strict=False
                )

            print(
                "Pretrained weights loaded successfully."
            )

            if len(missing_keys) > 0:

                print(
                    "Missing keys:",
                    len(missing_keys)
                )

            if len(unexpected_keys) > 0:

                print(
                    "Unexpected keys:",
                    len(unexpected_keys)
                )

            self.net_G.to(
                self.device
            )

            # Make sure training mode is enabled
            self.net_G.train()

        # --------------------------------------------------
        # Train from scratch
        # --------------------------------------------------

        else:

            print(
                'Training from scratch...'
            )

            self.net_G.to(
                self.device
            )

        print("\n")


    # ======================================================
    # TIMER
    # ======================================================

    def _timer_update(self):

        self.global_step = (
            self.epoch_id -
            self.epoch_to_start
        ) * self.steps_per_epoch + self.batch_id

        self.timer.update_progress(
            (self.global_step + 1) /
            self.total_steps
        )

        est = self.timer.estimated_remaining()

        imps = (
            (self.global_step + 1) *
            self.batch_size /
            self.timer.get_stage_elapsed()
        )

        return imps, est


    # ======================================================
    # VISUALIZATION
    # ======================================================

    def _visualize_pred(self):

        pred = torch.argmax(
            self.G_final_pred,
            dim=1,
            keepdim=True
        )

        pred_vis = pred * 255

        return pred_vis


    # ======================================================
    # SAVE CHECKPOINT
    # ======================================================

    def _save_checkpoint(
        self,
        ckpt_name
    ):

        torch.save(

            {
                'epoch_id':
                    self.epoch_id,

                'best_val_acc':
                    self.best_val_acc,

                'best_epoch_id':
                    self.best_epoch_id,

                'model_G_state_dict':
                    self.net_G.state_dict(),

                'optimizer_G_state_dict':
                    self.optimizer_G.state_dict(),

                'exp_lr_scheduler_G_state_dict':
                    self.exp_lr_scheduler_G.state_dict(),
            },

            os.path.join(
                self.checkpoint_dir,
                ckpt_name
            )
        )


    # ======================================================
    # LR SCHEDULER
    # ======================================================

    def _update_lr_schedulers(self):

        self.exp_lr_scheduler_G.step()


    # ======================================================
    # METRIC
    # ======================================================

    def _update_metric(self):

        target = (
            self.batch['L']
            .to(self.device)
            .detach()
        )

        G_pred = (
            self.G_final_pred
            .detach()
        )

        G_pred = torch.argmax(
            G_pred,
            dim=1
        )

        current_score = \
            self.running_metric.update_cm(

                pr=G_pred.cpu().numpy(),

                gt=target.cpu().numpy()
            )

        return current_score


    # ======================================================
    # BATCH LOGGING
    # ======================================================

    def _collect_running_batch_states(
        self
    ):

        running_acc = \
            self._update_metric()

        m = len(
            self.dataloaders['train']
        )

        if self.is_training is False:

            m = len(
                self.dataloaders['val']
            )

        imps, est = \
            self._timer_update()

        if np.mod(
            self.batch_id,
            100
        ) == 1:

            message = \
                'Is_training: %s. [%d,%d][%d,%d], ' \
                'imps: %.2f, est: %.2fh, ' \
                'G_loss: %.5f, running_mf1: %.5f\n' % (

                    self.is_training,

                    self.epoch_id,

                    self.max_num_epochs - 1,

                    self.batch_id,

                    m,

                    imps * self.batch_size,

                    est,

                    self.G_loss.item(),

                    running_acc
                )

            self.logger.write(
                message
            )

        if np.mod(
            self.batch_id,
            500
        ) == 1:

            vis_input = \
                utils.make_numpy_grid(
                    de_norm(
                        self.batch['A']
                    )
                )

            vis_input2 = \
                utils.make_numpy_grid(
                    de_norm(
                        self.batch['B']
                    )
                )

            vis_pred = \
                utils.make_numpy_grid(
                    self._visualize_pred()
                )

            vis_gt = \
                utils.make_numpy_grid(
                    self.batch['L']
                )

            vis = np.concatenate(
                [
                    vis_input,
                    vis_input2,
                    vis_pred,
                    vis_gt
                ],
                axis=0
            )

            vis = np.clip(
                vis,
                a_min=0.0,
                a_max=1.0
            )

            file_name = os.path.join(

                self.vis_dir,

                'istrain_' +
                str(self.is_training) +
                '_' +
                str(self.epoch_id) +
                '_' +
                str(self.batch_id) +
                '.jpg'
            )

            plt.imsave(
                file_name,
                vis
            )


    # ======================================================
    # EPOCH STATES
    # ======================================================

    def _collect_epoch_states(self):

        scores = \
            self.running_metric.get_scores()

        self.epoch_acc = \
            scores['mf1']

        self.logger.write(

            'Is_training: %s. Epoch %d / %d, '
            'epoch_mF1= %.5f\n'
            % (

                self.is_training,

                self.epoch_id,

                self.max_num_epochs - 1,

                self.epoch_acc
            )
        )

        message = ''

        for k, v in scores.items():

            message += \
                '%s: %.5f ' % (
                    k,
                    v
                )

        self.logger.write(
            message + '\n'
        )

        self.logger.write(
            '\n'
        )


    # ======================================================
    # CHECKPOINT UPDATE
    # ======================================================

    def _update_checkpoints(self):

        # --------------------------------------------------
        # Save latest model
        # --------------------------------------------------

        self._save_checkpoint(
            ckpt_name='last_ckpt.pt'
        )

        self.logger.write(

            'Latest model updated. '
            'Epoch_acc=%.4f, '
            'Historical_best_acc=%.4f '
            '(at epoch %d)\n'

            % (

                self.epoch_acc,

                self.best_val_acc,

                self.best_epoch_id
            )
        )

        self.logger.write(
            '\n'
        )

        # --------------------------------------------------
        # Save best model
        # --------------------------------------------------

        if self.epoch_acc > self.best_val_acc:

            self.best_val_acc = \
                self.epoch_acc

            self.best_epoch_id = \
                self.epoch_id

            self._save_checkpoint(
                ckpt_name='best_ckpt.pt'
            )

            self.logger.write(
                '*' * 10 +
                'Best model updated!\n'
            )

            self.logger.write(
                '\n'
            )


    # ======================================================
    # TRAINING CURVE
    # ======================================================

    def _update_training_acc_curve(self):

        self.TRAIN_ACC = np.append(
            self.TRAIN_ACC,
            [self.epoch_acc]
        )

        np.save(

            os.path.join(
                self.checkpoint_dir,
                'train_acc.npy'
            ),

            self.TRAIN_ACC
        )


    # ======================================================
    # VALIDATION CURVE
    # ======================================================

    def _update_val_acc_curve(self):

        self.VAL_ACC = np.append(
            self.VAL_ACC,
            [self.epoch_acc]
        )

        np.save(

            os.path.join(
                self.checkpoint_dir,
                'val_acc.npy'
            ),

            self.VAL_ACC
        )


    # ======================================================
    # CLEAR METRIC CACHE
    # ======================================================

    def _clear_cache(self):

        self.running_metric.clear()


    # ======================================================
    # FORWARD PASS
    # ======================================================

    def _forward_pass(
        self,
        batch
    ):

        self.batch = batch

        img_in1 = \
            batch['A'].to(
                self.device
            )

        img_in2 = \
            batch['B'].to(
                self.device
            )

        self.G_pred = \
            self.net_G(
                img_in1,
                img_in2
            )

        # --------------------------------------------------
        # Multi-scale inference
        # --------------------------------------------------

        if self.multi_scale_infer == "True":

            self.G_final_pred = \
                torch.zeros(
                    self.G_pred[-1].size()
                ).to(
                    self.device
                )

            for pred in self.G_pred:

                if pred.size(2) != \
                   self.G_pred[-1].size(2):

                    self.G_final_pred = \
                        self.G_final_pred + \
                        F.interpolate(

                            pred,

                            size=self.G_pred[-1].size(2),

                            mode="nearest"
                        )

                else:

                    self.G_final_pred = \
                        self.G_final_pred + pred

            self.G_final_pred = \
                self.G_final_pred / \
                len(self.G_pred)

        else:

            self.G_final_pred = \
                self.G_pred[-1]


    # ======================================================
    # BACKWARD PASS
    # ======================================================

    def _backward_G(self):

        gt = \
            self.batch['L'].to(
                self.device
            ).float()

        # --------------------------------------------------
        # Multi-scale training
        # --------------------------------------------------

        if self.multi_scale_train == "True":

            i = 0

            temp_loss = 0.0

            for pred in self.G_pred:

                if pred.size(2) != \
                   gt.size(2):

                    temp_loss = \
                        temp_loss + \
                        self.weights[i] * \
                        self._pxl_loss(

                            pred,

                            F.interpolate(

                                gt,

                                size=pred.size(2),

                                mode="nearest"
                            )
                        )

                else:

                    temp_loss = \
                        temp_loss + \
                        self.weights[i] * \
                        self._pxl_loss(

                            pred,
                            gt
                        )

                i += 1

            self.G_loss = temp_loss

        else:

            self.G_loss = \
                self._pxl_loss(
                    self.G_pred[-1],
                    gt
                )

        self.G_loss.backward()


    # ======================================================
    # TRAIN
    # ======================================================

    def train_models(self):

        self._load_checkpoint()

        # --------------------------------------------------
        # Epoch loop
        # --------------------------------------------------

        for self.epoch_id in range(
            self.epoch_to_start,
            self.max_num_epochs
        ):

            # ==================================================
            # TRAINING
            # ==================================================

            self._clear_cache()

            self.is_training = True

            self.net_G.train()

            total = len(
                self.dataloaders['train']
            )

            self.logger.write(
                'lr: %0.7f\n \n'
                %
                self.optimizer_G
                .param_groups[0]['lr']
            )

            # --------------------------------------------------
            # Batch loop
            # --------------------------------------------------

            for self.batch_id, batch in tqdm(

                enumerate(
                    self.dataloaders['train'],
                    0
                ),

                total=total
            ):

                self._forward_pass(
                    batch
                )

                # Clear gradients

                self.optimizer_G.zero_grad()

                # Backpropagation

                self._backward_G()

                # Update weights

                self.optimizer_G.step()

                # Logging

                self._collect_running_batch_states()

                self._timer_update()

            # --------------------------------------------------
            # Training epoch statistics
            # --------------------------------------------------

            self._collect_epoch_states()

            self._update_training_acc_curve()

            self._update_lr_schedulers()

            # ==================================================
            # VALIDATION
            # ==================================================

            self.logger.write(
                'Begin evaluation...\n'
            )

            self._clear_cache()

            self.is_training = False

            self.net_G.eval()

            # --------------------------------------------------
            # Validation batches
            # --------------------------------------------------

            for self.batch_id, batch in enumerate(

                self.dataloaders['val'],
                0
            ):

                with torch.no_grad():

                    self._forward_pass(
                        batch
                    )

                self._collect_running_batch_states()

            # --------------------------------------------------
            # Validation statistics
            # --------------------------------------------------

            self._collect_epoch_states()

            # --------------------------------------------------
            # Checkpoints
            # --------------------------------------------------

            self._update_val_acc_curve()

            self._update_checkpoints()