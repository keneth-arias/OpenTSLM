"""CPU-only fixtures for data selection, predictions and checkpoint adapters."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace as NS
import unittest

import numpy as np

from evaluation.opentslm.arfbench.protocol import SPLIT, covered, load_manifest, materialize, question
from evaluation.opentslm.arfbench.replay import DECODE, compare, probe_manifest, read_run, score


def fixture(task_id, incident, answer="cpu"):
    sources = []
    for name, values in [("cpu", [0, None, 8]), ("mem", [0, 1, 2])]:
        array = np.array(values, dtype=float)
        signal = NS(
            name="10s",
            n_values=3,
            spec=NS(modality="time_series"),
            to_arrow=lambda a=array: NS(to_numpy=lambda **kw: a),
        )
        sources.append(NS(name=name, signals=[signal]))
    record = NS(
        id=task_id,
        annotations=[NS(key="incident_id", value=incident)],
        sources=sources,
        signals=[s.signals[0] for s in sources],
    )
    return NS(
        id=task_id,
        inputs=[record],
        annotations=[NS(key="task_category", value="Anomaly Identification")],
        prompt="cpu or mem?",
        label_set=NS(labels=[NS(name=s) for s in ["cpu", "mem"]]),
        targets=[answer],
    )


class ProtocolTest(unittest.TestCase):
    def test_materialize(self):
        ids = dict(train=[1], validation=[], test=[2])
        tasks = [fixture(1, "a"), fixture(2, "b", "mem")]
        m = materialize(tasks, ids, str, max_series=1)
        self.assertEqual(m, materialize(tasks[::-1], ids, str, max_series=1))
        r, excluded = m["rows"]
        self.assertEqual(m["traces"][r["series"][0]]["values"], [0, None, 8])
        self.assertTrue(excluded["exclusion"])
        self.assertTrue(covered("x:1,y:2 and x:3,y:4", ["x:1,y:2", "x:3,y:4"]))
        self.assertFalse(covered("x:12", ["x:1"]))
        with TemporaryDirectory() as root:
            path = Path(root) / "manifest"
            path.write_text(json.dumps(m))
            self.assertEqual(load_manifest(path), m)
            path.write_text(json.dumps({**m, "max_series": 32}))
            with self.assertRaisesRegex(ValueError, "modified"):
                load_manifest(path)
        with self.assertRaisesRegex(ValueError, "overlaps"):
            materialize([tasks[0], fixture(2, "a")], ids, str)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            materialize([*tasks, tasks[1]], ids, str)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            materialize(tasks, {**ids, "test": [1]}, str)
        with self.assertRaisesRegex(ValueError, "missing from dataset"):
            materialize(tasks[:1], ids, str)
        self.assertEqual(
            {k: len(v) for k, v in json.loads(SPLIT.read_text()).items()}, {"train": 471, "validation": 68, "test": 135}
        )

    def test_failure_and_imbalanced_replay(self):
        from evaluation.opentslm.arfbench.run import evaluate

        m = probe_manifest("missing")
        config = dict(manifest_sha256=m["sha256"], split="missing", device="cpu", decoding=DECODE)

        class Fixture:
            def eval(self):
                pass

            def generate(self, series, prompt, *, max_new_tokens):
                self.prompt = prompt
                if len(series[0].values) == 2:
                    raise ValueError("intentional fixture error")
                return "None of the above" if 0 in series[0].values else "Answer: NO."

        with TemporaryDirectory() as root:
            fixture_model = Fixture()
            report = evaluate(fixture_model, m, config, Path(root))
            self.assertIn("Does the series", fixture_model.prompt)
            self.assertEqual(report["retained"], 3)
            self.assertEqual(report["scored"], 2)
            self.assertEqual(report["failures"], 1)
            self.assertEqual(report["unparseable"], 1)
            self.assertEqual(report["accuracy"], 1 / 3)
            self.assertEqual(report["per_category"]["missing"]["balanced_accuracy"], 0.25)
            run = read_run(root)
            self.assertEqual(report, score(m, *run)[0])
            better = [{**p, "output": r["answer"], "error": None} for p, r in zip(run[1], m["rows"])]
            self.assertEqual(compare(m, run, (config, better))["second_only"], 2)
            with self.assertRaisesRegex(ValueError, "exactly once"):
                compare(m, run, (config, better[:-1]))
            with self.assertRaisesRegex(ValueError, "decoding"):
                compare(m, run, ({**config, "decoding": {}}, better))

    def test_swap_controls(self):
        m = probe_manifest("swap")
        for row in m["rows"]:
            signals = [m["traces"][k] for k in row["series"]]
            names = ["cpu", "memory"]
            self.assertEqual([t["name"] for t in signals], names if row["pair"] == 0 else names[::-1])
            anomaly = next(t for t in signals if None in t["values"])
            self.assertEqual(anomaly["name"], row["answer"])
            self.assertFalse(anomaly["observed"][1])
            self.assertIn(20, anomaly["values"])
            self.assertEqual(row["options"], names)
        settings = dict(split="swap", manifest_sha256=m["sha256"])
        preds = [dict(id=r["id"], question=question(r), output=r["answer"]) for r in m["rows"]]
        preds[-1]["output"] = "unknown"
        self.assertEqual(score(m, settings, preds)[0]["fraction_pairs_both_correct"], 0.5)


class LegacyTest(unittest.TestCase):
    def test_legacy_prompt_only_and_gap(self):
        from evaluation.opentslm.arfbench.backends import Legacy, SeriesInput
        from unittest.mock import Mock

        backend = Mock()
        backend.generate.return_value = ["cpu"]
        result = Legacy(backend).generate([SeriesInput([0, None, 2], name="cpu")], "which?", max_new_tokens=2)
        self.assertEqual(result, "cpu")
        batch = backend.generate.call_args.args[0]
        self.assertNotIn("answer", batch[0])
        self.assertEqual(batch[0]["post_prompt"], "which?")
        self.assertEqual(batch[0]["time_series"].tolist(), [[-1, 0, 1, 0]])
        self.assertIn("mean 1.0000", batch[0]["time_series_text"][0])

    def test_legacy_checkpoint_roundtrip(self):
        import inspect
        from importlib.util import find_spec

        if find_spec("open_flamingo") is None:
            self.skipTest("requires benchmark legacy open-flamingo 2.0.1")
        import open_flamingo
        import torch
        from tokenizers import Tokenizer, models, pre_tokenizers
        from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
        from opentslm.model.llm.OpenTSLMFlamingo import OpenTSLMFlamingo
        from evaluation.opentslm.arfbench.backends import Legacy, SeriesInput, load

        if "gradient_checkpointing" not in inspect.signature(open_flamingo.Flamingo).parameters:
            self.skipTest("requires benchmark legacy open-flamingo 2.0.1")
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "backbone"
            vocab = {"[UNK]": 0, "[EOS]": 1, "cpu": 2, "which": 3}
            raw = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
            raw.pre_tokenizer = pre_tokenizers.Whitespace()
            tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw, unk_token="[UNK]", eos_token="[EOS]")
            tokenizer.save_pretrained(base)
            LlamaForCausalLM(
                LlamaConfig(
                    vocab_size=len(vocab),
                    hidden_size=16,
                    intermediate_size=32,
                    num_attention_heads=2,
                    num_key_value_heads=2,
                    num_hidden_layers=1,
                    eos_token_id=1,
                )
            ).save_pretrained(base)
            source = Legacy(OpenTSLMFlamingo(device="cpu", llm_id=str(base)))
            checkpoint = root / "save"
            source.save_pretrained(checkpoint)
            args = NS(
                checkpoint=str(checkpoint),
                revision=None,
                backbone=str(base),
                backbone_revision=None,
                device="cpu",
                backend="legacy",
            )
            restored, settings = load(args)
            self.assertGreater(settings["restored_parameter_counts"]["cross_attention"], 0)
            example = [SeriesInput([0, None, 2], name="cpu")]
            self.assertIsInstance(restored.generate(example, "which", max_new_tokens=2), str)
            losses = restored.loss(example, "which", "cpu", reduction="none")
            self.assertEqual(losses.shape, (2,))
            self.assertTrue(torch.isfinite(losses).all())
            file = checkpoint / "model_checkpoint.pt"
            state = torch.load(file, weights_only=True)
            del state["llm"]["vision_encoder.patch_embed.weight"]
            torch.save(state, file)
            with self.assertRaisesRegex(ValueError, "lacks"):
                load(args)
