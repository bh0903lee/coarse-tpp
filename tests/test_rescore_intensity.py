"""Re-scoring of EasyTPP intensity models: batch conventions and the quadrature against a closed form."""
import math

import numpy as np
import torch

from coarse_tpp import rescore_intensity as W


def cfg(model_id):
    return {"model_config": dict(hidden_size=8, time_emb_size=4, num_layers=1, num_heads=1, num_event_types=2,
                                 num_event_types_pad=3, event_pad_index=2, model_id=model_id)}


def seqs():
    rng = np.random.default_rng(0)
    out = []
    for n in (5, 3, 7):
        dt = np.round(rng.exponential(1.0, n), 1); dt[0] = 0.0
        dt = np.where(dt == 0, 1e-6, dt)
        out.append((np.cumsum(dt), dt, rng.integers(0, 2, n)))
    return out


def test_batch_follows_the_easytpp_tokenizer_conventions():
    ts, dts, ks, non_pad, attn = W.make_batch(seqs(), 2, "cpu")
    assert ts.shape == (3, 7) and int(non_pad.sum()) == 15 and bool(attn[0, 0, 1]) and not bool(attn[0, 1, 0])
    assert all(ks[1, 3:] == 2)


def test_rmtpp_scores_match_the_closed_form_and_the_interval_limit():
    torch.manual_seed(0)
    m = W.build_model("RMTPP", cfg("RMTPP"), torch.device("cpu"))
    b = W.make_batch(seqs(), 2, "cpu")
    o = W.event_scores(m, "RMTPP", b, 1e-4, 64, "cpu")
    mask = o["mask"] & ~o["tie"]
    # tiny resolution: interval score -> density + log delta on non-tie events
    assert torch.allclose(o["interval"][mask], o["floor_1e-6"][mask] + math.log(1e-4), atol=1e-3)
    res = W.score_split(m, "RMTPP", seqs(), 2, 0.1, 32, "cpu")
    assert res["n"] == 12 and math.isfinite(res["interval"]) and res["floor_1e-10_nonfinite"] == 0


def test_all_four_models_build_and_score():
    for mid in W.MODELS:
        torch.manual_seed(0)
        m = W.build_model(mid, cfg(mid), torch.device("cpu"))
        res = W.score_split(m, mid, seqs(), 2, 0.1, 8, "cpu")
        assert math.isfinite(res["floor_1e-6"]) and math.isfinite(res["interval"]), mid
