from lightning.pytorch.callbacks import LearningRateMonitor

from asymdsd.data import StampaAsymDSDDataModule
from asymdsd.models.asymdsd import TraingingMode
from asymdsd.models.asymdsd_pqdt_fab_packed import (
    PQDTPackedFusedAttnBlockAsymDSD,
)
from asymdsd.run.cli import TrainerCLI


def get_multi_crop_config(multi_crop_config, training_mode):
    if training_mode == TraingingMode.MASK:
        return None
    return multi_crop_config


LINKED_ARGS = [
    ("data.init_args.batch_size", "model.batch_size"),
    ("model.encoder_config.embed_dim", "model.projection_head_config.in_dim"),
    ("trainer.max_epochs", "model.max_epochs"),
    ("trainer.max_steps", "model.max_steps"),
]

TRAINER_DEFAULTS = {
    "callbacks": LearningRateMonitor(logging_interval="step", log_weight_decay=True),
}


def cli_main():
    TrainerCLI(
        model_class=PQDTPackedFusedAttnBlockAsymDSD,
        datamodule_class=StampaAsymDSDDataModule,
        linked_args_list=LINKED_ARGS,
        trainer_defaults=TRAINER_DEFAULTS,
        add_optim_key=True,
        subclass_mode_data=True,
    )


if __name__ == "__main__":
    cli_main()
