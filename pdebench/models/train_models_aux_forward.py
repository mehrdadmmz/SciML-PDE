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
        from fno_aux.fno_train_aux import run_training as run_training_FNO
        logger.info("FNO")
        run_training_FNO(
            if_training=cfg.args.if_training,
            if_downsample=cfg.args.if_downsample,
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
            learning_rate_share=cfg.args.learning_rate_share,
            learning_rate_fc2=cfg.args.learning_rate_fc2,
            num_aux_samples=cfg.args.num_aux_samples,
            auxiliary_weight=cfg.args.auxiliary_weight,
            # accumulation_steps=cfg.args.accumulation_steps,
            scheduler=cfg.args.scheduler,
            scheduler_step=cfg.args.scheduler_step,
            scheduler_gamma=cfg.args.scheduler_gamma,
            model_update=cfg.args.model_update,
            model_flmn=cfg.args.model_flmn,
            base_path=cfg.args.base_path,
            aux_path=cfg.args.aux_path,
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
