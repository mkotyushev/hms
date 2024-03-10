import logging
import torch
import timm
from lightning import LightningModule
from typing import Any, Dict, Optional, Union, Literal
from torch import Tensor
import torch.nn.functional as F
from lightning.pytorch.cli import instantiate_class
from torchmetrics import Metric
from lightning.pytorch.utilities import grad_norm

from .hms_classifier import HmsClassifier
from src.data.constants import N_CLASSES
from src.utils.utils import state_norm, patch_first_conv
from src.utils.mechanic import mechanize
from src.data.constants import LABEL_COLS_ORDERED, CONFUSION_MATRIX


logger = logging.getLogger(__name__)


class BaseModule(LightningModule):
    def __init__(
        self, 
        optimizer_init: Optional[Dict[str, Any]] = None,
        lr_scheduler_init: Optional[Dict[str, Any]] = None,
        pl_lrs_cfg: Optional[Dict[str, Any]] = None,
        finetuning: Optional[Dict[str, Any]] = None,
        log_norm_verbose: bool = False,
        lr_layer_decay: Union[float, Dict[str, float]] = 1.0,
        n_bootstrap: int = 1000,
        skip_nan: bool = False,
        prog_bar_names: Optional[list] = None,
        mechanize: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.metrics = None
        self.cat_metrics = None

        self.configure_metrics()

    def compute_loss_preds(self, batch, *args, **kwargs):
        """Compute losses and predictions."""

    def configure_metrics(self):
        """Configure task-specific metrics."""

    def bootstrap_metric(self, probas, targets, metric: Metric):
        """Calculate metric on bootstrap samples."""
        return None

    @staticmethod
    def check_batch_dims(batch):
        assert all(map(lambda x: len(x) == len(batch['eeg']), batch)), \
            f'All entities in batch must have the same length, got ' \
            f'{list(map(len, batch))}'

    def remove_nans(self, y, y_pred):
        nan_mask = torch.isnan(y_pred)
        
        if nan_mask.ndim > 1:
            nan_mask = nan_mask.any(dim=1)
        
        if nan_mask.any():
            if not self.hparams.skip_nan:
                raise ValueError(
                    f'Got {nan_mask.sum()} / {nan_mask.shape[0]} nan values in update_metrics. '
                    f'Use skip_nan=True to skip them.'
                )
            logger.warning(
                f'Got {nan_mask.sum()} / {nan_mask.shape[0]} nan values in update_metrics. '
                f'Dropping them & corresponding targets.'
            )
            y_pred = y_pred[~nan_mask]
            y = y[~nan_mask]
        return y, y_pred

    def extract_targets_and_probas_for_metric(self, preds, batch):
        """Extract preds and targets from batch.
        Could be overriden for custom batch / prediction structure.
        """
        y, y_pred = batch['label'].detach(), preds.detach().float()
        y, y_pred = self.remove_nans(y, y_pred)
        y_pred = torch.softmax(y_pred, dim=1)
        return y, y_pred

    def update_metrics(self, span, preds, batch):
        """Update train metrics."""
        y, y_proba = self.extract_targets_and_probas_for_metric(preds, batch)
        self.cat_metrics[span]['probas'].update(y_proba)
        self.cat_metrics[span]['targets'].update(y)

    def on_train_epoch_start(self) -> None:
        """Called in the training loop at the very beginning of the epoch."""
        # Unfreeze all layers if freeze period is over
        if self.hparams.finetuning is not None:
            # TODO change to >= somehow
            if self.current_epoch == self.hparams.finetuning['unfreeze_before_epoch']:
                self.unfreeze()

    def unfreeze_only_selected(self):
        """
        Unfreeze only layers selected by 
        model.finetuning.unfreeze_layer_names_*.
        """
        if self.hparams.finetuning is not None:
            for name, param in self.named_parameters():
                selected = False

                if 'unfreeze_layer_names_startswith' in self.hparams.finetuning:
                    selected = selected or any(
                        name.startswith(pattern) 
                        for pattern in self.hparams.finetuning['unfreeze_layer_names_startswith']
                    )

                if 'unfreeze_layer_names_contains' in self.hparams.finetuning:
                    selected = selected or any(
                        pattern in name
                        for pattern in self.hparams.finetuning['unfreeze_layer_names_contains']
                    )
                logger.info(f'Param {name}\'s requires_grad == {selected}.')
                param.requires_grad = selected

    def training_step(self, batch, batch_idx, **kwargs):
        total_loss, losses, preds = self.compute_loss_preds(batch, **kwargs)
        for loss_name, loss in losses.items():
            self.log(
                f'train_loss_{loss_name}', 
                loss,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                batch_size=batch['eeg'].shape[0],
            )
        self.update_metrics('train_metrics', preds, batch)

        # Handle nan in loss
        has_nan = False
        if torch.isnan(total_loss):
            has_nan = True
            logger.warning(
                f'Loss is nan at epoch {self.current_epoch} '
                f'step {self.global_step}.'
            )
        for loss_name, loss in losses.items():
            if torch.isnan(loss):
                has_nan = True
                logger.warning(
                    f'Loss {loss_name} is nan at epoch {self.current_epoch} '
                    f'step {self.global_step}.'
                )
        if has_nan:
            return None
        
        return total_loss
    
    def validation_step(self, batch: Tensor, batch_idx: int, dataloader_idx: Optional[int] = None, **kwargs) -> Tensor:
        total_loss, losses, preds = self.compute_loss_preds(batch, **kwargs)
        assert dataloader_idx is None or dataloader_idx == 0, 'Only one val dataloader is supported.'
        for loss_name, loss in losses.items():
            self.log(
                f'val_loss_{loss_name}', 
                loss,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                add_dataloader_idx=False,
                batch_size=batch['eeg'].shape[0],
            )
        self.update_metrics('val_metrics', preds, batch)
        return total_loss

    def predict_step(self, batch: Tensor, batch_idx: int, dataloader_idx: Optional[int] = None, **kwargs) -> Tensor:
        _, _, preds = self.compute_loss_preds(batch, **kwargs)
        return preds

    def log_metrics_and_reset(
        self, 
        prefix, 
        on_step=False, 
        on_epoch=True, 
        prog_bar_names=None,
        reset=True,
    ):
        # Get metric span: train or val
        span = None
        if prefix == 'train':
            span = 'train_metrics'
        elif prefix in ['val', 'val_ds']:
            span = 'val_metrics'
        
        # Get concatenated preds and targets
        # and reset them
        probas, targets = \
            self.cat_metrics[span]['probas'].compute().cpu(),  \
            self.cat_metrics[span]['targets'].compute().cpu()
        if reset:
            self.cat_metrics[span]['probas'].reset()
            self.cat_metrics[span]['targets'].reset()

        # Calculate and log metrics
        for name, metric in self.metrics.items():
            metric_value = None
            if prefix == 'val_ds':  # bootstrap
                if self.hparams.n_bootstrap > 0:
                    metric_value = self.bootstrap_metric(probas[:, 1], targets, metric)
                else:
                    logger.warning(
                        f'prefix == val_ds but n_bootstrap == 0. '
                        f'No bootstrap metrics will be calculated '
                        f'and logged.'
                    )
            else:
                metric.update(probas[:, 1], targets)
                metric_value = metric.compute()
                metric.reset()
            
            prog_bar = False
            if prog_bar_names is not None:
                prog_bar = (name in prog_bar_names)

            if metric_value is not None:
                self.log(
                    f'{prefix}_{name}',
                    metric_value,
                    on_step=on_step,
                    on_epoch=on_epoch,
                    prog_bar=prog_bar,
                )

    def on_train_epoch_end(self) -> None:
        """Called in the training loop at the very end of the epoch."""
        if self.metrics is None:
            return
        assert self.cat_metrics is not None

        self.log_metrics_and_reset(
            'train',
            on_step=False,
            on_epoch=True,
            prog_bar_names=self.hparams.prog_bar_names,
            reset=True,
        )
    
    def on_validation_epoch_end(self) -> None:
        """Called in the validation loop at the very end of the epoch."""
        if self.metrics is None:
            return
        assert self.cat_metrics is not None

        self.log_metrics_and_reset(
            'val',
            on_step=False,
            on_epoch=True,
            prog_bar_names=self.hparams.prog_bar_names,
            reset=False,
        )
        self.log_metrics_and_reset(
            'val_ds',
            on_step=False,
            on_epoch=True,
            prog_bar_names=self.hparams.prog_bar_names,
            reset=True,
        )

    def get_lr_decayed(self, lr, layer_index, layer_name):
        """
        Get lr decayed by 
            - layer index as (self.hparams.lr_layer_decay ** layer_index) if
              self.hparams.lr_layer_decay is float 
              (useful e. g. when new parameters are in classifer head)
            - layer name as self.hparams.lr_layer_decay[layer_name] if
              self.hparams.lr_layer_decay is dict
              (useful e. g. when pretrained parameters are at few start layers 
              and new parameters are the most part of the model)
        """
        if isinstance(self.hparams.lr_layer_decay, dict):
            for key in self.hparams.lr_layer_decay:
                if layer_name.startswith(key):
                    return lr * self.hparams.lr_layer_decay[key]
            return lr
        elif isinstance(self.hparams.lr_layer_decay, float):
            if self.hparams.lr_layer_decay == 1.0:
                return lr
            else:
                return lr * (self.hparams.lr_layer_decay ** layer_index)


    def build_parameter_groups(self):
        """Get parameter groups for optimizer."""
        names, params = list(zip(*self.named_parameters()))
        num_layers = len(params)
        
        if self.hparams.lr_layer_decay == 1.0:
            grouped_parameters = [
                {
                    'params': params, 
                    'lr': self.hparams.optimizer_init['init_args']['lr']
                }
            ]
        else:
            grouped_parameters = [
                {
                    'params': param, 
                    'lr': self.get_lr_decayed(
                        self.hparams.optimizer_init['init_args']['lr'], 
                        num_layers - layer_index - 1,
                        name
                    )
                } for layer_index, (name, param) in enumerate(self.named_parameters())
            ]
        
        logger.info(
            f'Number of layers: {num_layers}, '
            f'min lr: {names[0]}, {grouped_parameters[0]["lr"]}, '
            f'max lr: {names[-1]}, {grouped_parameters[-1]["lr"]}'
        )

        return grouped_parameters

    def configure_optimizer(self):
        if not self.hparams.mechanize:
            optimizer = instantiate_class(args=self.build_parameter_groups(), init=self.hparams.optimizer_init)
            return optimizer
        else:
            # similar to instantiate_class, but with mechanize
            args, init = self.build_parameter_groups(), self.hparams.optimizer_init
            kwargs = init.get("init_args", {})
            if not isinstance(args, tuple):
                args = (args,)
            class_module, class_name = init["class_path"].rsplit(".", 1)
            module = __import__(class_module, fromlist=[class_name])
            args_class = getattr(module, class_name)
            
            optimizer = mechanize(args_class)(*args, **kwargs)
            
            return optimizer

    def configure_lr_scheduler(self, optimizer):
        # Convert milestones from total persents to steps
        # for PiecewiceFactorsLRScheduler
        if (
            'PiecewiceFactorsLRScheduler' in self.hparams.lr_scheduler_init['class_path'] and
            self.hparams.pl_lrs_cfg['interval'] == 'step'
        ):
            total_steps = len(self.trainer.fit_loop._data_source.dataloader()) * self.trainer.max_epochs
            grad_accum_steps = self.trainer.accumulate_grad_batches
            self.hparams.lr_scheduler_init['init_args']['milestones'] = [
                int(milestone * total_steps / grad_accum_steps) 
                for milestone in self.hparams.lr_scheduler_init['init_args']['milestones']
            ]
        
        scheduler = instantiate_class(args=optimizer, init=self.hparams.lr_scheduler_init)
        scheduler = {
            "scheduler": scheduler,
            **self.hparams.pl_lrs_cfg,
        }

        return scheduler

    def configure_optimizers(self):
        optimizer = self.configure_optimizer()
        if self.hparams.lr_scheduler_init is None:
            return optimizer

        scheduler = self.configure_lr_scheduler(optimizer)

        return [optimizer], [scheduler]

    def on_before_optimizer_step(self, optimizer):
        """Log gradient norms."""
        # Compute the 2-norm for each layer
        # If using mixed precision, the gradients are already unscaled here
        norms = grad_norm(self, norm_type=2)
        if self.hparams.log_norm_verbose:
            self.log_dict(norms)
        else:
            if 'grad_2.0_norm_total' in norms:
                self.log('grad_2.0_norm_total', norms['grad_2.0_norm_total'])

        norms = state_norm(self, norm_type=2)
        if self.hparams.log_norm_verbose:
            self.log_dict(norms)
        else:
            if 'state_2.0_norm_total' in norms:
                self.log('state_2.0_norm_total', norms['state_2.0_norm_total'])


class HmsModule(BaseModule):
    def __init__(
        self,
        model: str = 'hms_classifier',
        model_kwargs=None,
        weight_by_n_voters: bool = False,
        label_smoothing: bool = False,
        lr=None,
        **base_kwargs,
    ):
        super().__init__(**base_kwargs)
        self.save_hyperparameters()

        if model_kwargs is None:
            model_kwargs = dict()

        if model == 'hms_classifier':
            self.model = HmsClassifier(
                n_classes=N_CLASSES,
                input_dim_s=100,
                num_patches_s=1200,
                input_dim_e=200,
                num_patches_e=1000,
                **model_kwargs,
            )
        else:
            in_chans = model_kwargs.pop('in_chans', 3)
            self.model = timm.create_model(model, num_classes=N_CLASSES, **model_kwargs)
            patch_first_conv(self.model, in_chans)
        
    def compute_loss_preds(self, batch, *args, **kwargs):
        weight_by_n_voters = kwargs.get('weight_by_n_voters', False)
        label_smoothing = kwargs.get('label_smoothing', False)

        if self.hparams.model == 'hms_classifier':
            if self.hparams.use == 'all':
                x_s, x_e = batch['spectrogram'], batch['eeg']
            elif self.hparams.use == 'spectrogram':
                x_s, x_e = batch['spectrogram'], None
            elif self.hparams.use == 'eeg':
                x_s, x_e = None, batch['eeg']
            preds = self.model(x_s, x_e)
        else:
            preds = self.model(batch['image'])
        target = batch['label'] / batch['label'].sum(1)[:, None]
        if label_smoothing:
            target = target @ torch.from_numpy(CONFUSION_MATRIX).to(target.dtype).to(target.device)
        log_preds = F.log_softmax(preds, dim=1)
        kld = F.kl_div(
            log_preds, 
            target,
            log_target=False,
            reduction='none',
        ).sum(1)

        if weight_by_n_voters:
            n_voters = torch.from_numpy(batch['meta'][LABEL_COLS_ORDERED].values.sum(1)).to(kld.device).float()
            kld = (kld * n_voters).sum() / n_voters.sum()
        else:
            # Same as with reduction='batchmean'
            kld = kld.sum() / kld.shape[0]

        losses = {
            'kld': kld
        }
        return sum(losses.values()), losses, log_preds

    def training_step(self, batch, batch_idx, **kwargs):
        train_kwargs = {
            'weight_by_n_voters': self.hparams.weight_by_n_voters,
            'label_smoothing': self.hparams.label_smoothing,
        }
        return super().training_step(batch, batch_idx, **{**kwargs, **train_kwargs})

    def update_metrics(self, span, preds, batch):
        """Update train metrics."""