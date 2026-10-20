"""Regression for Transformers 5 set-valued FSDP2 wrapping metadata."""

import unittest
from types import SimpleNamespace
from unittest.mock import call, patch

import torch
from omegaconf import OmegaConf

from verl.utils.fsdp_utils import apply_fsdp2


class DecoderBlock(torch.nn.Linear):
    def __init__(self):
        super().__init__(2, 2)


class ToyTransformer(torch.nn.Module):
    def __init__(self, layer_names):
        super().__init__()
        self.config = SimpleNamespace(tie_word_embeddings=True)
        self._no_split_modules = layer_names
        self.embedding = torch.nn.Embedding(4, 2)
        self.layers = torch.nn.ModuleList([DecoderBlock(), DecoderBlock()])
        self.head = torch.nn.Linear(2, 4)


class FSDP2WrapTests(unittest.TestCase):
    def test_set_and_sequence_metadata_wrap_the_same_layers(self):
        for names in (
            {"DecoderBlock"},
            frozenset({"DecoderBlock"}),
            ["DecoderBlock"],
            ("DecoderBlock",),
            OmegaConf.create(["DecoderBlock"]),
        ):
            with self.subTest(metadata_type=type(names).__name__):
                model = ToyTransformer(names)
                kwargs = {"reshard_after_forward": True}
                # Exercise real layer selection, mocking only distributed sharding.
                with patch("verl.utils.fsdp_utils.fully_shard") as shard:
                    apply_fsdp2(model, kwargs, {})
                self.assertEqual(
                    shard.call_args_list,
                    [call(model.layers[0], **kwargs), call(model.layers[1], **kwargs), call(model, **kwargs)],
                )

    def test_explicit_string_policy_overrides_model_metadata(self):
        model = ToyTransformer({"UnusedLayer"})
        with patch("verl.utils.fsdp_utils.fully_shard") as shard:
            apply_fsdp2(model, {}, {"wrap_policy": {"transformer_layer_cls_to_wrap": "DecoderBlock"}})
        self.assertEqual(shard.call_args_list, [call(model.layers[0]), call(model.layers[1]), call(model)])

    def test_invalid_metadata_fails_before_distributed_sharding(self):
        for names in (None, [], set(), {None}, ["DecoderBlock", None], [""]):
            with self.subTest(names=names), patch("verl.utils.fsdp_utils.fully_shard") as shard:
                with self.assertRaisesRegex(ValueError, "nonempty collection"):
                    apply_fsdp2(ToyTransformer(names), {}, {})
                shard.assert_not_called()


if __name__ == "__main__":
    unittest.main()
