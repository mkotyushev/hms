from typing import Dict, Optional, Union
from lightning import Trainer
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger


###################################################################
########################## General Utils ##########################
###################################################################

class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        """Add argument links to parser.

        Example:
            parser.link_arguments("data.init_args.img_size", "model.init_args.img_size")
        """
        return



class TrainerWandb(Trainer):
    """Hotfix for wandb logger saving config & artifacts to project root dir
    and not in experiment dir."""
    @property
    def log_dir(self) -> Optional[str]:
        """The directory for the current experiment. Use this to save images to, etc...

        .. code-block:: python

            def training_step(self, batch, batch_idx):
                img = ...
                save_img(img, self.trainer.log_dir)
        """
        if len(self.loggers) > 0:
            if isinstance(self.loggers[0], WandbLogger):
                dirpath = self.loggers[0]._experiment.dir
            elif not isinstance(self.loggers[0], TensorBoardLogger):
                dirpath = self.loggers[0].save_dir
            else:
                dirpath = self.loggers[0].log_dir
        else:
            dirpath = self.default_root_dir

        dirpath = self.strategy.broadcast(dirpath)
        return dirpath


class ModelCheckpointNoSave(ModelCheckpoint):
    def best_epoch(self) -> int:
        # exmple: epoch=10-step=1452.ckpt
        return int(self.best_model_path.split('=')[-2].split('-')[0])
    
    def ith_epoch_score(self, i: int) -> Optional[float]:
        # exmple: epoch=10-step=1452.ckpt
        ith_epoch_filepath_list = [
            filepath 
            for filepath in self.best_k_models.keys()
            if f'epoch={i}-' in filepath
        ]
        
        # Not found
        if not ith_epoch_filepath_list:
            return None
    
        ith_epoch_filepath = ith_epoch_filepath_list[-1]
        return self.best_k_models[ith_epoch_filepath]

    def _save_checkpoint(self, trainer: Trainer, filepath: str) -> None:
        self._last_global_step_saved = trainer.global_step

        # notify loggers
        if trainer.is_global_zero:
            for logger in trainer.loggers:
                logger.after_save_checkpoint(proxy(self))


class TempSetContextManager:
    def __init__(self, obj, attr, value):
        self.obj = obj
        self.attr = attr
        self.value = value

    def __enter__(self):
        self.old_value = getattr(self.obj, self.attr)
        setattr(self.obj, self.attr, self.value)

    def __exit__(self, *args):
        setattr(self.obj, self.attr, self.old_value)



def state_norm(module: torch.nn.Module, norm_type: Union[float, int, str], group_separator: str = "/") -> Dict[str, float]:
    """Compute each state dict tensor's norm and their overall norm.

    The overall norm is computed over all tensor together, as if they
    were concatenated into a single vector.

    Args:
        module: :class:`torch.nn.Module` to inspect.
        norm_type: The type of the used p-norm, cast to float if necessary.
            Can be ``'inf'`` for infinity norm.
        group_separator: The separator string used by the logger to group
            the tensor norms in their own subfolder instead of the logs one.

    Return:
        norms: The dictionary of p-norms of each parameter's gradient and
            a special entry for the total p-norm of the tensor viewed
            as a single vector.
    """
    norm_type = float(norm_type)
    if norm_type <= 0:
        raise ValueError(f"`norm_type` must be a positive number or 'inf' (infinity norm). Got {norm_type}")

    norms = {
        f"state_{norm_type}_norm{group_separator}{name}": p.data.float().norm(norm_type)
        for name, p in module.state_dict().items()
        if not 'num_batches_tracked' in name
    }
    if norms:
        total_norm = torch.tensor(list(norms.values())).norm(norm_type)
        norms[f"state_{norm_type}_norm_total"] = total_norm
    return norms


###################################################################
##################### CV ##########################################
###################################################################

# Segmentation

# https://github.com/bnsreenu/python_for_microscopists/blob/master/
# 229_smooth_predictions_by_blending_patches/smooth_tiled_predictions.py
def _spline_window(window_size, power=2):
    """
    Squared spline (power=2) window function:
    https://www.wolframalpha.com/input/?i=y%3Dx**2,+y%3D-(x-2)**2+%2B2,+y%3D(x-4)**2,+from+y+%3D+0+to+2
    """
    intersection = int(window_size / 4)
    wind_outer = (abs(2*(scipy.signal.triang(window_size))) ** power)/2
    wind_outer[intersection:-intersection] = 0

    wind_inner = 1 - (abs(2*(scipy.signal.triang(window_size) - 1)) ** power)/2
    wind_inner[:intersection] = 0
    wind_inner[-intersection:] = 0

    wind = wind_inner + wind_outer
    wind = wind / np.average(wind)
    return wind


def _spline_window_2d(h, w, power=2):
    h_wind = _spline_window(h, power)
    w_wind = _spline_window(w, power)
    return h_wind[:, None] * w_wind[None, :]


def _unpatchify2d_avg(  # pylint: disable=too-many-locals
    patches: np.ndarray, imsize: Tuple[int, int], weight_mode='uniform',
) -> np.ndarray:
    assert len(patches.shape) == 4
    assert weight_mode in ['uniform', 'spline']

    i_h, i_w = imsize
    image = np.zeros(imsize, dtype=np.float32)
    weights = np.zeros(imsize, dtype=np.float32)

    n_h, n_w, p_h, p_w = patches.shape

    s_w = 0 if n_w <= 1 else (i_w - p_w) / (n_w - 1)
    s_h = 0 if n_h <= 1 else (i_h - p_h) / (n_h - 1)

    # The step size should be same for all patches, otherwise the patches are unable
    # to reconstruct into a image
    if int(s_w) != s_w:
        raise NonUniformStepSizeError(i_w, n_w, p_w, s_w)
    if int(s_h) != s_h:
        raise NonUniformStepSizeError(i_h, n_h, p_h, s_h)
    s_w = int(s_w)
    s_h = int(s_h)

    weight = 1  # uniform
    if weight_mode == 'spline':
        weight = _spline_window_2d(p_h, p_w, power=2)

    # For each patch, add it to the image at the right location
    for i in range(n_h):
        for j in range(n_w):
            image[i * s_h : i * s_h + p_h, j * s_w : j * s_w + p_w] += (patches[i, j] * weight)
            weights[i * s_h : i * s_h + p_h, j * s_w : j * s_w + p_w] += weight

    # Average
    weights = np.where(np.isclose(weights, 0.0), 1.0, weights)
    image /= weights

    image = image.astype(patches.dtype)

    return image, weights


class PredictionTargetPreviewAgg(nn.Module):
    """Aggregate prediction and target patches to images with downscaling."""
    def __init__(
        self, 
        preview_downscale: Optional[int] = 4, 
        metrics=None, 
        input_std=1, 
        input_mean=0, 
        fill_value=0,
        overlap_avg_weight_mode='uniform',
    ):
        super().__init__()
        self.preview_downscale = preview_downscale
        self.metrics = metrics
        self.previews = {}
        self.shapes = {}
        self.shapes_before_padding = {}
        self.input_std = input_std
        self.input_mean = input_mean
        self.fill_value = fill_value
        self.overlap_avg_weight_mode = overlap_avg_weight_mode

    def reset(self):
        # Note: metrics are reset in compute()
        self.previews = {}
        self.shapes = {}
        self.shapes_before_padding = {}

    def update(
        self, 
        arrays: Dict[str, torch.Tensor | np.ndarray],
        pathes: list[str], 
        patch_size: torch.LongTensor | np.ndarray,
        indices: torch.LongTensor | np.ndarray, 
        shape_patches: torch.LongTensor | np.ndarray,
        shape_original: torch.LongTensor | np.ndarray,
        shape_before_padding: torch.LongTensor,
    ):
        # To CPU & types
        for name in arrays:
            if isinstance(arrays[name], torch.Tensor):
                arrays[name] = arrays[name].cpu().numpy()
            
            if name == 'input':
                arrays[name] = ((arrays[name] * self.input_std + self.input_mean) * 255).astype(np.uint8)
            elif name == 'probas':
                arrays[name] = arrays[name].astype(np.float32)
            elif name == ['mask', 'target']:
                arrays[name] = arrays[name].astype(np.uint8)
            else:
                # Do not convert type
                pass

        if isinstance(indices, torch.Tensor):
            indices = indices.cpu().numpy()
        if isinstance(shape_patches, torch.Tensor):
            shape_patches = shape_patches.cpu().numpy()
        if isinstance(shape_before_padding, torch.Tensor):
            shape_before_padding = shape_before_padding.cpu().numpy()

        indices, shape_patches, shape_before_padding = \
            indices.astype(np.int64), \
            shape_patches.astype(np.int64), \
            shape_before_padding.astype(np.int64)
    
        # Place patches on the preview images
        B = arrays[list(arrays.keys())[0]].shape[0]
        for i in range(B):
            path = Path(pathes[i])
            path = str(path.relative_to(path.parent.parent))
            shape = [
                *shape_patches[i].tolist(),
                *patch_size,
            ]

            self.shapes[path] = shape_original[i].tolist()[:2]
            self.shapes_before_padding[path] = shape_before_padding[i].tolist()[:2]
            patch_index_w, patch_index_h = indices[i].tolist()

            for name, value in arrays.items():
                key = f'{name}|{path}'
                if key not in self.previews:
                    self.previews[key] = np.full(shape, fill_value=self.fill_value, dtype=arrays[name].dtype)
                    if name.startswith('probas'):
                        # Needed to calculate average from sum
                        # hack to not change dict size later, actually computed in compute()
                        self.previews[f'counts|{path}'] = None
                self.previews[key][patch_index_h, patch_index_w] = value[i]
    
    def compute(self):
        # Unpatchify
        for name in self.previews:
            path = name.split('|')[-1]
            shape_original = self.shapes[path]
            if name.startswith('probas'):
                # Average overlapping patches
                self.previews[name], counts = _unpatchify2d_avg(
                    self.previews[name], 
                    shape_original,
                    weight_mode=self.overlap_avg_weight_mode,
                )
                self.previews[name.replace('probas', 'counts')] = counts.astype(np.uint8)
            elif name.startswith('counts'):
                # Do nothing
                pass
            else:
                # Just unpatchify
                self.previews[name] = unpatchify(
                    self.previews[name], 
                    shape_original
                )

        # Zero probas out where mask is zero
        for name in self.previews:
            if name.startswith('probas'):
                mask = self.previews[name.replace('probas', 'mask')] == 0
                self.previews[name][mask] = 0

        # Crop to shape before padding
        for name in self.previews:
            path = name.split('|')[-1]
            shape_before_padding = self.shapes_before_padding[path]
            self.previews[name] = self.previews[name][
                :shape_before_padding[0], 
                :shape_before_padding[1],
            ]

        # Compute metrics if available
        metric_values = None
        if self.metrics is not None:
            preds, targets = [], []
            for name in self.previews:
                if name.startswith('probas'):
                    path = name.split('|')[-1]
                    mask = self.previews[f'mask|{path}'] > 0
                    pred = self.previews[name][mask].flatten()
                    target = self.previews[f'target|{path}'][mask].flatten()

                    preds.append(pred)
                    targets.append(target)
            preds = torch.from_numpy(np.concatenate(preds))
            targets = torch.from_numpy(np.concatenate(targets))

            metric_values = {}
            for metric_name, metric in self.metrics.items():
                metric.update(preds, targets)
                metric_values[metric_name] = metric.compute()
                metric.reset()
        
        # Downscale and get captions
        captions, previews = [], []
        for name, preview in self.previews.items():
            if self.preview_downscale is not None:
                preview = cv2.resize(
                    preview,
                    dsize=(0, 0),
                    fx=1 / self.preview_downscale, 
                    fy=1 / self.preview_downscale, 
                    interpolation=cv2.INTER_LINEAR, 
                )
            captions.append(name)
            previews.append(preview)

        return metric_values, captions, previews
    

class PredictionTargetPreviewGrid(nn.Module):
    """Aggregate prediction and target patches to images with downscaling."""
    def __init__(self, preview_downscale: int = 4, n_images: int = 4):
        super().__init__()
        self.preview_downscale = preview_downscale
        self.n_images = n_images
        self.previews = defaultdict(list)

    def reset(self):
        self.previews = defaultdict(list)

    def update(
        self, 
        input: torch.Tensor,
        probas: torch.Tensor, 
        target: torch.Tensor, 
        pathes: list[str],
    ):
        # Add images until grid is full
        for i in range(probas.shape[0]):
            path = '/'.join(pathes[i].split('/')[-2:])
            if len(self.previews[f'input_{path}']) < self.n_images:
                # Get preview images
                inp = F.interpolate(
                    input[i].float().unsqueeze(0),
                    scale_factor=1 / self.preview_downscale, 
                    mode='bilinear',
                    align_corners=False, 
                ).cpu()
                proba = F.interpolate(
                    probas[i].float().unsqueeze(0).unsqueeze(1),  # interpolate as (N, C, H, W)
                    scale_factor=1 / self.preview_downscale, 
                    mode='bilinear', 
                    align_corners=False, 
                ).cpu()
                targ = F.interpolate(
                    target[i].float().unsqueeze(0).unsqueeze(1),  # interpolate as (N, C, H, W)
                    scale_factor=1 / self.preview_downscale,
                    mode='bilinear',
                    align_corners=False, 
                ).cpu()

                self.previews[f'input_{path}'].append(inp)
                self.previews[f'proba_{path}'].append((proba * 255).byte())
                self.previews[f'target_{path}'].append((targ * 255).byte())
    
    def compute(self):
        captions = list(self.previews.keys())
        preview_grids = [
            make_grid(
                torch.cat(v, dim=0), 
                nrow=int(self.n_images ** 0.5)
            ).float()
            for v in self.previews.values()
        ]

        return captions, preview_grids
