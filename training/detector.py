"""Train-only wrapper for the frozen Ultralytics scratch and native EMA recipes."""

import hashlib
import json
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

import yolo11s_scratch as scratch
import yolo11s_native_tiles as native


class StopProfile(Exception):
    pass


class ForbiddenValidator:
    metrics = SimpleNamespace(keys=[])
    calls = 0

    def __call__(self, *args, **kwargs):
        self.calls += 1
        raise RuntimeError("No validation pass is defined in all-707 deployment training")


def steps_expected(images, epochs, warmup_epochs):
    n = (images + 3) // 4
    warm = max(round(warmup_epochs * n), 100)
    previous, updates = -1, 0
    for ix in range(epochs * n):
        accumulate = max(1, int(np.interp(ix, [0, warm], [1, 16]).round())) if ix <= warm else 16
        if ix - previous >= accumulate:
            updates += 1
            previous = ix
    return n, updates


def fit_detector(component, yaml_path, folder, parent_checkpoint, budget, device, profile, timer):
    with patch.object(scratch, "RUN", folder.parent):
        YOLO, architecture = scratch.setup()
    from ultralytics.data import utils as data_utils
    from ultralytics.models.yolo.segment.train import SegmentationTrainer
    from ultralytics.utils.torch_utils import strip_optimizer

    settings = scratch.read(scratch.CONFIG)["training"].copy()
    if component == "native":
        settings.update(scratch.read(native.CONFIG)["training"])
        if parent_checkpoint is None or not parent_checkpoint.is_file():
            raise ValueError("The exact own full-disk EMA is required before native continuation")
    elif parent_checkpoint is not None:
        raise ValueError("The scratch stage must start from a local random model configuration")
    if settings["epochs"] != budget["epochs"] or settings["amp"] or settings["batch"] != 4:
        raise ValueError("Selected detector recipe changed")
    settings["val"] = False  # Stock still calls validate() at final epoch, so override it explicitly.
    dataset_count = budget["batches_per_epoch"] * 4 - (4 - budget["tail"])
    totals, clock, schedule, epochs, proof = [], [], hashlib.sha256(), [], {}
    updates, best_calls, no_op_calls, tail_count = 0, 0, 0, 0
    gpu_start = timer()
    torch.manual_seed(settings["seed"])
    model = YOLO(str(parent_checkpoint) if parent_checkpoint else "yolo11s-seg.yaml", task="segment", verbose=False)
    if parent_checkpoint is None and (model.ckpt or model.ckpt_path is not None or model.model.yaml.get("scale") != "s"):
        raise ValueError("Unexpected non-scratch architecture or checkpoint")

    class AllFitTrainer(SegmentationTrainer):
        def get_dataloader(self, *args, mode="train", **kwargs):
            if mode == "val":
                # The library requires a startup loader for its abstract API.
                # This sentinel cannot iterate or expose annotation samples.
                return SimpleNamespace(dataset=None)
            return super().get_dataloader(*args, mode=mode, **kwargs)

        def get_validator(self):
            self.loss_names = "box_loss", "seg_loss", "cls_loss", "dfl_loss"
            return ForbiddenValidator()

        def validate(self):
            nonlocal no_op_calls
            no_op_calls += 1
            return {}, 0.0

        def preprocess_batch(self, batch):
            nonlocal tail_count
            ids = [str(x).replace("\\", "/").rsplit("/", 1)[-1] for x in batch["im_file"]]
            if len(ids) == budget["tail"] and budget["tail"] < 4:
                tail_count += 1
            totals.append(len(ids))
            schedule.update((" ".join(ids) + "\n").encode("ascii"))
            return super().preprocess_batch(batch)

        def optimizer_step(self):
            nonlocal updates
            super().optimizer_step()
            updates += 1

        def save_model(self):
            nonlocal best_calls
            if self.epoch != settings["epochs"] - 1:
                return
            super().save_model()
            if self.best.exists():
                best_calls += 1
                raise ValueError("Validation-selected best checkpoint was created")

        def final_eval(self):
            if self.best.exists() or not self.last.exists() or self.validator.calls:
                raise ValueError("Only the exact final EMA weights may be retained")
            if not strip_optimizer(self.last):
                raise ValueError("Could not strip the exact final EMA")

    def batch_complete(trainer):
        if profile:
            torch.cuda.synchronize(device)
            if not bool(torch.isfinite(trainer.loss).all()):
                raise ValueError("Nonfinite detector profile loss")
            clock.append(timer())
            if len(clock) == budget["profile_batches"]:
                raise StopProfile

    def epoch_complete(trainer):
        if not bool(torch.isfinite(trainer.loss).all()):
            raise ValueError("Nonfinite detector epoch")
        item = {"epoch": trainer.epoch + 1, "batches": len(totals), "updates": updates,
                "loss_shape": list(trainer.loss_items.shape),
                "last_losses": [float(v) for v in trainer.loss_items.detach().cpu().flatten()]}
        epochs.append(item)
        print(json.dumps(item, allow_nan=False), flush=True)

    def on_started(trainer):
        if len(trainer.train_loader.dataset) != dataset_count or len(trainer.train_loader) != budget["batches_per_epoch"]:
            raise ValueError("Effective exact all707 batch schedule changed")
        proof["init_classes"] = int(trainer.data["nc"])
        trainer.metrics = {}

    model.add_callback("on_train_batch_end", batch_complete)
    model.add_callback("on_train_epoch_end", epoch_complete)
    model.add_callback("on_train_start", on_started)
    with patch.object(data_utils, "check_font", lambda *_: None):
        try:
            model.train(trainer=AllFitTrainer, data=str(yaml_path), project=str(folder), name="engine",
                        device=device.index, **settings)
        except StopProfile:
            if not profile:
                raise
    trainer = model.trainer
    checkpoint = folder / "engine/weights/last.pt"
    if trainer.validator.calls or best_calls or trainer.best.exists():
        raise ValueError("Validation or best-checkpoint side effects are prohibited")
    if profile:
        if len(clock) != budget["profile_batches"] or checkpoint.exists():
            raise ValueError("Exact disposable detector profile schedule changed")
        # A private scratch-profile EMA is for native profiling only and will
        # never be accepted by the full native production guard.
        if component == "scratch":
            from copy import deepcopy

            state = deepcopy(trainer.ema.ema).half().cpu()
            proof_state = {"model": state, "train_args": vars(trainer.args)}
            torch.save(proof_state, folder / "native-profile-parent.pt")
    elif (len(epochs) != budget["epochs"] or len(totals) != budget["epochs"] * budget["batches_per_epoch"]
          or updates != budget["optimizer_updates"] or tail_count != (budget["epochs"] if budget["tail"] < 4 else 0)
          or sum(totals) != dataset_count * budget["epochs"] or not checkpoint.is_file()):
        raise ValueError("Final EMA receipt does not match the frozen epoch and optimizer counts")
    return {"architecture": architecture, "checkpoint": checkpoint, "budgets": budget,
            "effective_settings": settings, "observed_batches": len(totals),
            "observed_items": sum(totals), "optimizer_updates": updates, "tail_batches": tail_count,
            "validated_samples": 0, "validator_calls": trainer.validator.calls,
            "constant_terminal_validation_hook_calls": no_op_calls, "best_checkpoint": False,
            "schedule_sha256": schedule.hexdigest(), "epoch_history": epochs,
            "mean_steady_seconds": float(np.mean(np.diff(clock)[6:])) if profile else None,
            "profile_batch_seconds": np.diff(clock).tolist() if profile else None, **proof,
            "elapsed_internal_seconds": timer() - gpu_start}
