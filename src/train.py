import torch
import numpy as np

from train_config import PoseNetTrainConfig

from neural_pipeline import Trainer
from neural_pipeline.monitoring import LogMonitor
from neural_pipeline.utils.fsm import FileStructManager
from neural_pipeline.builtin.monitors.tensorboard import TensorboardMonitor


EPOCH_NUM = 20


def train():
    train_config = PoseNetTrainConfig()

    file_struct_manager = FileStructManager(base_dir=PoseNetTrainConfig.experiment_dir, is_continue=False)

    trainer = Trainer(train_config, file_struct_manager, torch.device('cuda'))
    trainer.set_epoch_num(EPOCH_NUM)

    tensorboard = TensorboardMonitor(file_struct_manager, is_continue=False)
    log = LogMonitor(file_struct_manager).write_final_metrics()
    trainer.monitor_hub.add_monitor(tensorboard).add_monitor(log)
    trainer.enable_best_states_saving(lambda: np.mean(train_config.val_stage.get_losses()))

    trainer.enable_lr_decaying(coeff=0.5, patience=10, target_val_clbk=lambda: np.mean(train_config.val_stage.get_losses()))
    trainer.add_on_epoch_end_callback(lambda: tensorboard.update_scalar('params/lr', trainer.data_processor().get_lr()))
    trainer.train()


if __name__ == "__main__":
    train()
