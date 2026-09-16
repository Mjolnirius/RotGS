import os
import re
import subprocess
from pathlib import Path


class ExperimentTracker:
    """Small optional W&B adapter used by the single-camera trainer."""

    RUN_ID_FILENAME = "wandb_run_id"

    def __init__(self, run=None, wandb_module=None, artifact_policy="none"):
        self._run = run
        self._wandb = wandb_module
        self.artifact_policy = artifact_policy

    @property
    def enabled(self):
        return self._run is not None

    @property
    def run_id(self):
        return self._run.id if self.enabled else None

    @classmethod
    def create(cls, args, output_path):
        if args.wandb_mode == "disabled":
            return cls()

        try:
            import wandb
        except ImportError as exc:
            raise RuntimeError(
                "W&B tracking was requested but wandb is not installed; "
                "run `uv add wandb` and try again"
            ) from exc

        output_path = os.path.abspath(output_path)
        run_id = args.wandb_run_id
        run_id_path = os.path.join(output_path, cls.RUN_ID_FILENAME)
        if run_id is None and args.start_checkpoint and os.path.isfile(run_id_path):
            run_id = Path(run_id_path).read_text(encoding="utf-8").strip() or None

        config = vars(args).copy()
        # cfg_args retains the full local path. W&B gets portable identifiers
        # instead of a machine-specific dataset location.
        config.pop("source_path", None)
        if config.get("start_checkpoint"):
            config["start_checkpoint"] = os.path.basename(config["start_checkpoint"])
        repository_root = Path(__file__).resolve().parents[1]
        try:
            config["git_commit"] = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=repository_root,
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
            config["git_dirty"] = bool(
                subprocess.check_output(
                    ["git", "status", "--porcelain"],
                    cwd=repository_root,
                    stderr=subprocess.DEVNULL,
                    text=True,
                ).strip()
            )
        except (OSError, subprocess.CalledProcessError):
            pass

        run = wandb.init(
            entity=args.wandb_entity,
            project=args.wandb_project,
            name=args.wandb_run_name or args.name,
            group=args.wandb_group,
            tags=args.wandb_tags or None,
            notes=args.wandb_notes,
            config=config,
            dir=output_path,
            mode=args.wandb_mode,
            id=run_id,
            force=args.wandb_mode == "online",
            resume="allow" if run_id else None,
            job_type="train",
            save_code=False,
            settings=wandb.Settings(x_disable_meta=True),
        )
        run.define_metric("iteration")
        run.define_metric("*", step_metric="iteration")
        Path(run_id_path).write_text(run.id + "\n", encoding="utf-8")
        print(f"W&B run: {run.url or run.id}")
        return cls(run, wandb, artifact_policy=args.wandb_artifacts)

    def image(self, image, caption=None):
        if not self.enabled:
            return None
        image = (
            image.detach()
            .cpu()
            .float()
            .nan_to_num(nan=0.0, posinf=1.0, neginf=0.0)
            .clamp(0.0, 1.0)
            .mul(255)
            .round()
            .byte()
        )
        return self._wandb.Image(image, caption=caption)

    def log(self, metrics, iteration):
        if not self.enabled:
            return
        payload = {"iteration": iteration}
        payload.update(metrics)
        self._run.log(payload)

    def get_summary(self, key, default=None):
        if not self.enabled:
            return default
        return self._run.summary.get(key, default)

    def set_summary(self, key, value):
        if self.enabled:
            self._run.summary[key] = value

    def log_model_artifact(self, model_path, iteration, aliases):
        if not self.enabled or self.artifact_policy == "none":
            return

        artifact = self._wandb.Artifact(
            name=f"rotgs-model-{self._run.id}",
            type="model",
            metadata={"iteration": iteration},
        )
        point_cloud_path = os.path.join(
            model_path, "point_cloud", f"iteration_{iteration}"
        )
        residual_path = os.path.join(
            model_path, "residual_predictor", f"iteration_{iteration}"
        )
        if os.path.isdir(point_cloud_path):
            artifact.add_dir(
                point_cloud_path,
                name=f"point_cloud/iteration_{iteration}",
            )
        if os.path.isdir(residual_path):
            artifact.add_dir(
                residual_path,
                name=f"residual_predictor/iteration_{iteration}",
            )
        for filename in ("cfg_args", "exposure.json"):
            path = os.path.join(model_path, filename)
            if os.path.isfile(path):
                artifact.add_file(path, name=filename)

        aliases = [re.sub(r"[^A-Za-z0-9_.-]+", "-", alias) for alias in aliases]
        self._run.log_artifact(artifact, aliases=aliases)

    def finish(self, exit_code=0):
        if self.enabled:
            self._run.finish(exit_code=exit_code)
