"""Exercise the workflow checkpoint against local Git and DVC remotes."""

import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml


def test_training_checkpoint():
    workflow = Path(__file__).resolve().parents[1] / ".github/workflows/train.yml"
    steps = {
        step["name"]: step["run"]
        for step in yaml.safe_load(workflow.read_text())["jobs"]["train"]["steps"]
        if "run" in step
    }
    python = shlex.quote(sys.executable)

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        origin = root / "origin"
        origin.mkdir()
        env = dict(
            os.environ,
            GITHUB_REF_NAME="train_retry",
            DVC_NO_ANALYTICS="true",
            DVC_SITE_CACHE_DIR=str(root / "site-cache"),
            XDG_CACHE_HOME=str(root / "cache"),
            XDG_CONFIG_HOME=str(root / "config"),
        )

        def run(cwd, *command, success=True):
            result = subprocess.run(
                command, cwd=cwd, env=env, text=True, capture_output=True, check=False
            )
            assert (result.returncode == 0) == success, result.stdout + result.stderr
            return result.stdout

        def step(cwd, name, success=True):
            # Use this test's DVC installation, without syncing the toy project.
            script = steps[name].replace("uv run dvc", f"{python} -m dvc")
            return run(
                cwd, "bash", "-e", "-o", "pipefail", "-c", script, success=success
            )

        def checkout(name):
            path = root / name
            run(
                root,
                "git",
                "clone",
                "-q",
                "--branch",
                "train_retry",
                str(origin),
                str(path),
            )
            return path

        run(origin, "git", "init", "-q", "-b", "train_retry")
        run(origin, "git", "config", "user.name", "Checkpoint test")
        run(origin, "git", "config", "user.email", "checkpoint@example.invalid")
        run(origin, sys.executable, "-m", "dvc", "init", "-q")
        run(
            origin,
            sys.executable,
            "-m",
            "dvc",
            "remote",
            "add",
            "-d",
            "local",
            str(root / "remote"),
        )
        (origin / "train.py").write_text(
            "from pathlib import Path\nfrom uuid import uuid4\n"
            "Path('model').write_text(Path('input').read_text() + str(uuid4()))\n"
        )
        (origin / "dvc.yaml").write_text(
            yaml.safe_dump(
                {
                    "stages": {
                        "fetch_model_input": {
                            "cmd": f"{python} -c \"from pathlib import Path; Path('input').write_text('data')\"",
                            "outs": ["input"],
                        },
                        "train_yolo_best": {
                            "cmd": f"{python} train.py",
                            "deps": ["input", "train.py"],
                            "outs": ["model"],
                        },
                        "report": {
                            "cmd": f'{python} -c "raise SystemExit(1)"',
                            "deps": ["model"],
                        },
                    }
                }
            )
        )
        run(origin, "git", "add", ".")
        run(origin, "git", "commit", "-qm", "Training inputs")

        first = checkout("first")
        step(first, "Restore the trained model checkpoint")  # No result branch yet.
        step(first, "Train")
        weights = (first / "model").read_text()
        step(first, "Push the trained model to DVC remote")
        step(first, "Publish the model pointer to the result branch")
        step(first, "Run the rest of the pipeline", success=False)

        retry = checkout("retry")  # Empty cache and original training commit.
        step(retry, "Restore the trained model checkpoint")
        assert (retry / "input").read_text() == "data"
        step(retry, "Train")
        assert (retry / "model").read_text() == weights

        # Even code outside DVC's declared dependencies must invalidate reuse.
        (origin / "training_config.py").write_text("BATCH_SIZE = 32\n")
        run(origin, "git", "add", "training_config.py")
        run(origin, "git", "commit", "-qm", "Changed training code")
        changed = checkout("changed")
        step(changed, "Restore the trained model checkpoint")
        assert not (changed / "model").exists()
        step(changed, "Train")
        assert (changed / "model").read_text() != weights

        run(origin, "git", "revert", "--no-edit", "HEAD")
        (root / "remote").rename(root / "unavailable-remote")
        missing = checkout("missing")
        step(missing, "Restore the trained model checkpoint", success=False)
        assert not (missing / "model").exists()
