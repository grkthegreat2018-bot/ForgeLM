"""ForgeGate mechanics tests — stub model, CPU only.

Covers: probe bundle version guard, route threshold, doom escalation,
convergence exit with min_conv guard, token accounting. The stub model
returns a fixed 4-tuple (logits, None, kv, hidden) so probe scores are
fully controlled by probe bias terms.
"""
import torch
import pytest

from forge.engine.gated import (GateConfig, GateProbes, GatedDecoder,
                                GATE_PROBES_VERSION)


class _Enc:
    def __init__(self, ids):
        self.input_ids = ids


class FakeTok:
    def __call__(self, s, add_special_tokens=True):
        if s == '<|im_end|>':
            return _Enc([999])
        if s == '<|endoftext|>':
            return _Enc([998])
        return _Enc([1] * (len(s) % 7 + 2))

    def decode(self, ids, skip_special_tokens=True):
        return 'x' * len(ids)


class StubModel(torch.nn.Module):
    """Always argmaxes token 5 (non-EOS); hidden = constant."""

    def __init__(self, d=8, vocab=1000):
        super().__init__()
        self.d, self.vocab = d, vocab

    def forward(self, idx, use_cache=False, return_hidden=False,
                past_key_values=None, attention_mask=None, **kw):
        B, L = idx.shape
        logits = torch.zeros(B, L, self.vocab)
        logits[..., 5] = 10.0
        hidden = torch.randn(B, L, self.d)
        return logits, None, [], hidden


def _probes(route_b, doom_b, conv_b, d=8):
    return GateProbes(
        torch.zeros(2 * d, 1), torch.tensor([route_b]),   # concat feat
        torch.zeros(d, 1), torch.tensor([doom_b]),
        torch.zeros(d + 1, 1), torch.tensor([conv_b]),    # h + pos feat
    )


def _decoder(route_b, doom_b, conv_b, **cfg_kw):
    cfg = GateConfig(direct_max=8, think_max=12, ans_max=4,
                     min_conv=3, **cfg_kw)
    model = StubModel()
    return GatedDecoder(model, FakeTok(), torch.device('cpu'),
                        _probes(route_b, doom_b, conv_b), cfg)


class TestProbeBundle:
    def test_version_guard(self, tmp_path):
        p = tmp_path / 'bad.pt'
        torch.save({'version': 99, 'route': {}, 'doom': {}, 'conv': {}},
                   p)
        with pytest.raises(ValueError, match='version'):
            GateProbes.load(p, torch.device('cpu'))

    def test_load_ok(self, tmp_path):
        p = tmp_path / 'ok.pt'
        torch.save({'version': GATE_PROBES_VERSION,
                    'route': {'w': torch.zeros(4, 1),
                              'b': torch.zeros(1)},
                    'doom': {'w': torch.zeros(4, 1),
                             'b': torch.zeros(1)},
                    'conv': {'w': torch.zeros(5, 1),
                             'b': torch.zeros(1)}}, p)
        pr = GateProbes.load(p, torch.device('cpu'))
        assert pr.route[0].shape == (4, 1)


class TestGatePaths:
    def test_think_route_when_p_easy_low(self):
        dec = _decoder(route_b=-10.0, doom_b=0.0, conv_b=-10.0)
        res = dec.generate('q')
        assert res.path == 'think'
        assert res.p_easy < 0.5
        assert res.tokens == 12            # think_max, no fire

    def test_direct_completes_when_clean(self):
        dec = _decoder(route_b=10.0, doom_b=-10.0, conv_b=-10.0)
        res = dec.generate('q')
        assert res.path == 'direct'
        assert res.tokens == 8             # direct_max

    def test_doom_escalates_to_think(self):
        dec = _decoder(route_b=10.0, doom_b=10.0, conv_b=-10.0)
        res = dec.generate('q')
        assert res.path == 'escalated'
        assert res.tokens == 2 + 12        # wasted doom steps + think_max

    def test_conv_exit_after_min_conv(self):
        dec = _decoder(route_b=-10.0, doom_b=0.0, conv_b=10.0)
        res = dec.generate('q')
        assert res.path == 'think+exit'
        # K=2 consecutive hits; min_conv=3 -> fires at step 4,
        # gen holds steps 0..4 -> 5 tokens + ans_max
        assert res.fire_pos == 4
        assert res.tokens == 5 + 4

    def test_escalated_exit_path(self):
        dec = _decoder(route_b=10.0, doom_b=10.0, conv_b=10.0)
        res = dec.generate('q')
        assert res.path == 'escalated+exit'
