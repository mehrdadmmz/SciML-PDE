
from __future__ import annotations

import logging

import hydra
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


@hydra.main(version_base="1.2", config_path="config", config_name="config_ns")
def main(cfg: DictConfig):
    if not hasattr(cfg, "dataset") or cfg.dataset not in cfg:
        raise ValueError("You must specify a dataset")

    selected_dataset = cfg.dataset  # Dynamically selected dataset
    train_subsample = cfg[selected_dataset].train_subsample  

    if cfg.args.model_name == "FNO":
        from fno.train import run_training as run_training_FNO

        logger.info("FNO")
        run_training_FNO(
            if_training=cfg.args.if_training,
            continue_training=cfg.args.continue_training,
            rollout_test=cfg.args.rollout_test,
            num_workers=cfg.args.num_workers,
            modes=cfg.args.modes,
            width=cfg.args.width,
            initial_step=cfg.args.initial_step,
            t_train=cfg.args.t_train,
            training_type=cfg.args.training_type,
            num_channels=cfg.args.num_channels,
            batch_size=cfg.args.batch_size,
            epochs=cfg.args.epochs,
            train_subsample=train_subsample,
            learning_rate=cfg.args.learning_rate,
            scheduler=cfg.args.scheduler,
            scheduler_step=cfg.args.scheduler_step,
            scheduler_gamma=cfg.args.scheduler_gamma,
            model_update=cfg.args.model_update,
            FNO_model_flmn=cfg.args.FNO_model_flmn,
            base_path=cfg.args.base_path,
            plot=cfg.args.plot,
            channel_plot=cfg.args.channel_plot,
            x_min=cfg.args.x_min,
            x_max=cfg.args.x_max,
            y_min=cfg.args.y_min,
            y_max=cfg.args.y_max,
            t_min=cfg.args.t_min,
            t_max=cfg.args.t_max,
        )
   

if __name__ == "__main__":
    main()
