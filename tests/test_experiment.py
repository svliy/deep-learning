import io
import json
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch
import yaml

from config import TrainConfig, load_config, save_config
from config.cli import parse_args
from experiment import save_training_results, training_session


class ExperimentTests(unittest.TestCase):
    def test_experiment_names_cannot_escape_output_root(self):
        for name in ("", " ", "../other", "/tmp/other", "a/b", "a\\b", ".", "..", None):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "experiment_name"):
                TrainConfig(experiment_name=name).validate()

    def test_experiment_name_from_yaml_can_be_overridden(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text("experiment_name: yaml_experiment\n", encoding="utf-8")
            self.assertEqual(load_config(path).experiment_name, "yaml_experiment")
            with patch("sys.argv", ["train.py", "--experiment-name", "cli_experiment"]):
                self.assertEqual(load_config(path, parse_args()).experiment_name, "cli_experiment")

    def test_runs_keep_results_and_logs_together(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            config = TrainConfig(experiment_name="实验一", output_dir=str(root / "results"))
            runs = []
            with redirect_stderr(io.StringIO()):
                for index in (1, 2):
                    with training_session(config) as (logger, run_dir):
                        runs.append(run_dir)
                        save_config(config, run_dir / "config.yaml")
                        save_training_results(
                            {"epoch": index, "weight": torch.tensor([index])},
                            [{"epoch": index}],
                            run_dir,
                            is_best=True,
                        )
                        logger.info("运行记录 %d", index)
                    self.assertEqual(logger.handlers, [])
            self.assertNotEqual(runs[0], runs[1])
            for index, run_dir in enumerate(runs, start=1):
                self.assertEqual(run_dir.parent, root / "results/实验一")
                log = (run_dir / "train.log").read_text(encoding="utf-8")
                for message in (
                    "程序开始",
                    "实验名称：实验一",
                    f"运行记录 {index}",
                    "程序结束：正常结束",
                ):
                    self.assertIn(message, log)
                snapshot = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
                self.assertEqual(snapshot["experiment_name"], "实验一")
                self.assertEqual(
                    json.loads((run_dir / "history.json").read_text())[0]["epoch"], index
                )
                self.assertEqual(torch.load(run_dir / "best.pt", weights_only=True)["epoch"], index)
                self.assertTrue((run_dir / "last.pt").is_file())


if __name__ == "__main__":
    unittest.main()
