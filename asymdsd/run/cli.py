import os
from functools import wraps
from typing import Sequence

import torch
from lightning.pytorch.cli import LightningArgumentParser, LightningCLI

from asymdsd.callbacks import SaveModelHparams
from asymdsd.components.optimizer_spec import OptimizerSpec
from asymdsd.components.utils import compile_model as compile_model_fn
from asymdsd.defaults import DEFAULT_OPTIMIZER
from asymdsd.loggers import setup_logger

_original_torch_load = torch.load
torch.load = lambda *args, **kwargs: _original_torch_load(
    *args, **{**kwargs, "weights_only": kwargs.get("weights_only", False)}
)


def compile_model(func):
    @wraps(func)
    def wrapper(self, model, **kwargs):
        subcommand = self.config.subcommand
        config_kwargs = self.config.as_dict()
        compile_kwargs = config_kwargs[subcommand]["compile"]
        compile_kwargs.pop("__path__", None)
        model = compile_model_fn(model, **compile_kwargs)

        return func(self, model, **kwargs)

    return wrapper


class TrainerCLI(LightningCLI):
    def __init__(
        self,
        linked_args_list: Sequence[tuple[str | tuple[str], str]] | None = None,
        default_optimizer: OptimizerSpec = DEFAULT_OPTIMIZER,
        add_optim_key: bool = False,
        **kwargs,
    ) -> None:
        if "save_config_callback" not in kwargs:
            kwargs["save_config_callback"] = SaveModelHparams
        self.linked_args_list = linked_args_list or []
        self.default_optimizer = default_optimizer
        self.add_optim_key = add_optim_key
        self._setup_logger()
        super().__init__(**kwargs)

    def _setup_logger(self) -> None:
        info = warning = level = None
        if "LOG_LEVEL" in os.environ:
            level = os.environ["LOG_LEVEL"]
        if "INFO_LOG_FILE" in os.environ:
            info = os.environ["INFO_LOG_FILE"]
        if "WARNING_LOG_FILE" in os.environ:
            warning = os.environ["WARNING_LOG_FILE"]

        setup_logger(level=level, info_output=info, warn_output=warning)

    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        for linked_args in self.linked_args_list:
            parser.link_arguments(*linked_args, apply_on="parse")

        if self.add_optim_key:
            parser.add_subclass_arguments(
                OptimizerSpec,
                "optim",
                default=self.default_optimizer,
                instantiate=True,
            )
            parser.link_arguments("optim", "model.optimizer", apply_on="instantiate")

        parser.add_function_arguments(
            compile_model_fn, skip=set(["model"]), nested_key="compile"
        )

    @compile_model
    def fit(self, model, **kwargs):
        self.trainer.fit(model, **kwargs)
