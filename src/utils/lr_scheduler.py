import numpy as np
from torch.optim.lr_scheduler import LRScheduler


class ExpLRSchedulerPiece:
    def __init__(self, start_lr, stop_lr, decay=0.2):
        self.start_lr = start_lr
        self.scale = (start_lr - stop_lr) / (start_lr - start_lr * np.exp(-1.0 / decay))
        self.decay = decay

    def __call__(self, pct):
        # Parametrized so that in 0.0 it is start_lr and 
        # in 1.0 it is stop_lr
        # shift -> scale -> shift
        return \
            (
                self.start_lr * np.exp(-pct / self.decay) - 
                self.start_lr
            ) * self.scale + \
        self.start_lr
    

class ConstLRSchedulerPiece:
    def __init__(self, start_lr):
        self.start_lr = start_lr

    def __call__(self, pct):
        return self.start_lr
    

class LinearLRSchedulerPiece:
    def __init__(self, start_lr, stop_lr):
        self.start_lr = start_lr
        self.stop_lr = stop_lr

    def __call__(self, pct):
        return self.start_lr + pct * (self.stop_lr - self.start_lr)
    

class CosineLRSchedulerPiece:
    def __init__(self, start_lr, stop_lr):
        self.start_lr = start_lr
        self.stop_lr = stop_lr

    def __call__(self, pct):
        return self.stop_lr + (self.start_lr - self.stop_lr) * (1 + np.cos(np.pi * pct)) / 2


class PiecewiceFactorsLRScheduler(LRScheduler):
    """
    Piecewise learning rate scheduler.

    Each piece operates between two milestones. The first milestone is always 0.
    Given percent of the way through the current piece, piece yields the learning rate.
    Last piece is continued indefinitely for epoch > last milestone.
    """
    def __init__(self, optimizer, milestones, pieces, last_epoch=-1):
        assert len(milestones) - 1 == len(pieces)
        assert milestones[0] == 0
        assert all(milestones[i] < milestones[i + 1] for i in range(len(milestones) - 1))

        self.milestones = milestones
        self.pieces = pieces
        self._current_piece_index = 0

        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if (
            not self._current_piece_index == len(self.pieces) - 1 and 
            self.last_epoch > self.milestones[self._current_piece_index + 1]
        ):
            self._current_piece_index += 1

        pct = (
            self.last_epoch - 
            self.milestones[self._current_piece_index]
        ) / (
            self.milestones[self._current_piece_index + 1] - 
            self.milestones[self._current_piece_index]
        )

        return [
            self.pieces[self._current_piece_index](pct) * base_lr
            for base_lr in self.base_lrs
        ]
