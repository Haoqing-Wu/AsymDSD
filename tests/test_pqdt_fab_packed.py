from types import SimpleNamespace

import pytest
import torch


def _skip_if_local_pqdt_deps_unavailable():
    try:
        from pytorch3d.loss import chamfer_distance  # noqa: F401
        from pytorch3d.ops import sample_farthest_points  # noqa: F401
    except Exception as exc:
        pytest.skip(f"Local PQDT dependencies are not usable: {exc}")


def _import_or_skip(import_fn):
    try:
        return import_fn()
    except Exception as exc:
        pytest.skip(f"AsymDSD import dependencies are not usable: {exc}")


def _tiny_model():
    (
        RandomPatchMasking,
        ProjectionHeadConfig,
        TransformerEncoderConfig,
        PQDTPackedFusedAttnBlockAsymDSD,
    ) = _import_or_skip(
        lambda: (
            __import__(
                "asymdsd.components",
                fromlist=["RandomPatchMasking"],
            ).RandomPatchMasking,
            __import__(
                "asymdsd.layers",
                fromlist=["ProjectionHeadConfig"],
            ).ProjectionHeadConfig,
            __import__(
                "asymdsd.layers",
                fromlist=["TransformerEncoderConfig"],
            ).TransformerEncoderConfig,
            __import__(
                "asymdsd.models.asymdsd_pqdt_fab_packed",
                fromlist=["PQDTPackedFusedAttnBlockAsymDSD"],
            ).PQDTPackedFusedAttnBlockAsymDSD,
        )
    )

    model = PQDTPackedFusedAttnBlockAsymDSD(
        max_epochs=1,
        steps_per_epoch=1,
        batch_size=2,
        mask_generator=RandomPatchMasking(mask_ratio=0.5, multi_mask=4),
        encoder_config=TransformerEncoderConfig(
            embed_dim=32,
            num_heads=2,
            num_layers=1,
        ),
        predictor_config=None,
        projection_head_config=ProjectionHeadConfig(
            out_dim=64,
            hidden_dim=32,
            bottleneck_dim=16,
        ),
        pqdt_in_chans=32,
        pqdt_stem_enc_attn=("ge_attn", "attn"),
        pqdt_enc_attn=("ge_attn", "attn"),
        pqdt_dec_attn=("ge_attn", "attn"),
        pqdt_transdown_fps=(16, 12),
        pqdt_transdown_dims=(16, 32),
        pqdt_transdown_num_heads=(1, 2),
        pqdt_transdown_sa_depth=(1, 1),
        pqdt_transdown_k=(4, 4),
        pqdt_num_pseudo=8,
        pqdt_num_query=8,
        pqdt_total_epochs=1,
        pqdt_in_q=True,
        pqdt_up_factors=(1, 2),
        pqdt_up_n_knn=4,
        pqdt_cd_weight=0.1,
        semantic_visible_mask_ratio=0.5,
        semantic_masked_mask_ratio=0.5,
        geometric_reverse_mask_ratio=0.5,
        geometric_halfspace_mask_ratio=0.5,
        attn_num_layers=1,
        mask_probability=1.0,
        add_unmasked_global_cls=False,
        koleo_loss_weight=None,
        gradient_checkpointing=False,
    )
    model.setup()
    model.on_fit_start()
    return model


def test_pqdt_tail_forward_queries_tiny_shape():
    _skip_if_local_pqdt_deps_unavailable()
    PQDTTail = _import_or_skip(
        lambda: (
            __import__(
                "asymdsd.models.pqdt_tail",
                fromlist=["PQDTTail"],
            ).PQDTTail
        )
    )

    tail = PQDTTail(
        embed_dim=32,
        num_heads=2,
        enc_attn=("ge_attn", "attn"),
        dec_attn=("ge_attn", "attn"),
        num_pseudo=8,
        num_query=8,
        total_epochs=1,
        in_q=True,
    )
    out = tail.forward_queries(
        points=torch.randn(2, 32, 3),
        coor_c=torch.randn(2, 3, 12),
        x1=torch.randn(2, 12, 32),
        query_seed=torch.randn(2, 3, 5),
        current_epoch=0,
    )

    assert out.shape == (2, 5, 32)


def test_pqdt_packed_fab_tiny_training_step_smoke():
    _skip_if_local_pqdt_deps_unavailable()
    model = _tiny_model()
    output = model.training_step({"points": torch.randn(2, 32, 3)}, 0)

    assert torch.isfinite(output["loss"])
    assert output["patch_loss"] is not None
    assert output["cls_loss"] is not None
    assert torch.isfinite(output["pqdt_cd_loss"])
    assert output["pqdt_reconstructions"][-1].shape == (2, 16, 3)


def test_pqdt_reconstruction_gradients_are_isolated():
    _skip_if_local_pqdt_deps_unavailable()
    model = _tiny_model()
    model.zero_grad(set_to_none=True)

    cd_loss, _ = model._pqdt_reconstruction_loss(torch.randn(2, 32, 3))
    cd_loss.backward()

    assert any(
        param.grad is not None and torch.any(param.grad != 0)
        for param in model.up_layers.parameters()
    )
    assert all(param.grad is None for param in model.student.parameters())
    assert all(param.grad is None for param in model.teacher.parameters())


def test_pqdt_component_export_uses_pqdt_loadable_keys():
    _skip_if_local_pqdt_deps_unavailable()
    model = _tiny_model()
    state_dict = model.pqdt_component_state_dict(pqdt_loadable=True)

    assert any(key.startswith("stem_encoder.") for key in state_dict)
    assert any(key.startswith("transformer.decoder_2.") for key in state_dict)
    assert any(key.startswith("up_layers.") for key in state_dict)
    assert not any(key.startswith("teacher.") for key in state_dict)
    assert not any(key.startswith("student.") for key in state_dict)


def test_pqdt_export_shapes_match_external_pqdt_when_available():
    _skip_if_local_pqdt_deps_unavailable()
    try:
        import sys

        sys.path.insert(0, "/home/ubuntu/code/PQDT")
        from models.pq_completion import PQCompletionModel
    except Exception as exc:
        pytest.skip(f"External PQDT imports are not usable: {exc}")

    model = _tiny_model()
    exported = model.pqdt_component_state_dict(pqdt_loadable=True, cpu=False)
    cfg = SimpleNamespace(
        name="pqdt",
        trans_dim=32,
        num_heads=2,
        enc_attn=("ge_attn", "attn"),
        dec_attn=("ge_attn", "attn"),
        num_pseudo=8,
        num_queries=8,
        tau0=1.0,
        r_sph=0.8,
        in_q=True,
        up_n_knn=4,
        up_radius=1.0,
        up_factors=(1, 2),
        up_interpolate="three",
        up_attn_channel=True,
        stem_encoder=SimpleNamespace(
            in_chans=32,
            enc_attn=("ge_attn", "attn"),
            transdown_fps=(16, 12),
            transdown_dims=(16, 32),
            transdown_num_heads=(1, 2),
            transdown_sa_depth=(1, 1),
            transdown_k=(4, 4),
        ),
    )
    try:
        pqdt_model = PQCompletionModel(cfg, total_epochs=1)
    except Exception as exc:
        pytest.skip(f"External PQDT model construction is not usable: {exc}")

    stem = {
        key.removeprefix("stem_encoder."): value
        for key, value in exported.items()
        if key.startswith("stem_encoder.")
    }
    transformer = {
        key.removeprefix("transformer."): value
        for key, value in exported.items()
        if key.startswith("transformer.")
    }
    up_layers = {
        key.removeprefix("up_layers."): value
        for key, value in exported.items()
        if key.startswith("up_layers.")
    }

    assert all(
        key in pqdt_model.stem_encoder.state_dict()
        and pqdt_model.stem_encoder.state_dict()[key].shape == value.shape
        for key, value in stem.items()
    )
    pqdt_model.transformer.load_state_dict(transformer, strict=True)
    pqdt_model.up_layers.load_state_dict(up_layers, strict=True)
