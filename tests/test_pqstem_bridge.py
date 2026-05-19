import pytest
import torch


def _skip_if_local_stem_deps_unavailable():
    try:
        from pytorch3d.loss import chamfer_distance  # noqa: F401
        from pytorch3d.ops import sample_farthest_points  # noqa: F401
    except Exception as exc:
        pytest.skip(f"Local PQStem dependencies are not usable: {exc}")


def _import_or_skip(import_fn):
    try:
        return import_fn()
    except Exception as exc:
        pytest.skip(f"AsymDSD import dependencies are not usable: {exc}")


def test_pqstem_point_encoder_tiny_forward():
    _skip_if_local_stem_deps_unavailable()
    PQStemPatches, PQStemPointEncoder = _import_or_skip(
        lambda: (
            __import__(
                "asymdsd.models.pq_stem_point_encoder",
                fromlist=["PQStemPatches"],
            ).PQStemPatches,
            __import__(
                "asymdsd.models.pq_stem_point_encoder",
                fromlist=["PQStemPointEncoder"],
            ).PQStemPointEncoder,
        )
    )

    encoder = PQStemPointEncoder(
        in_chans=32,
        embed_dim=32,
        num_heads=2,
        enc_attn=("ge_attn", "attn"),
        transdown_fps=(16, 12),
        transdown_dims=(16, 32),
        transdown_num_heads=(1, 2),
        transdown_sa_depth=(1, 1),
        transdown_k=(4, 4),
    )
    points = torch.randn(2, 32, 3)
    centers = encoder.compute_centers(points)
    tokens = encoder.patch_embedding(PQStemPatches(points=points, centers=[centers]))
    output = encoder.transformer_encoder_forward(
        tokens.embeddings,
        tokens.pos_embeddings,
        token_centers=tokens.centers,
        return_attention=True,
    )

    assert output.patch_features.shape == (2, 12, 32)
    assert output.cls_features.shape == (2, 32)
    assert len(output.attn_weights) == 1
    assert output.attn_weights[0].shape == (2, 2, 13, 13)


def test_pqstem_packed_fab_tiny_training_step_smoke():
    _skip_if_local_stem_deps_unavailable()
    (
        RandomPatchMasking,
        ProjectionHeadConfig,
        TransformerDecoderConfig,
        TransformerEncoderConfig,
        PQStemPackedFusedAttnBlockAsymDSD,
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
                fromlist=["TransformerDecoderConfig"],
            ).TransformerDecoderConfig,
            __import__(
                "asymdsd.layers",
                fromlist=["TransformerEncoderConfig"],
            ).TransformerEncoderConfig,
            __import__(
                "asymdsd.models.asymdsd_pqstem_fab_packed",
                fromlist=["PQStemPackedFusedAttnBlockAsymDSD"],
            ).PQStemPackedFusedAttnBlockAsymDSD,
        )
    )

    model = PQStemPackedFusedAttnBlockAsymDSD(
        max_epochs=1,
        steps_per_epoch=1,
        batch_size=2,
        mask_generator=RandomPatchMasking(mask_ratio=0.5, multi_mask=4),
        encoder_config=TransformerEncoderConfig(
            embed_dim=32,
            num_heads=2,
            num_layers=1,
        ),
        predictor_config=TransformerDecoderConfig(
            embed_dim=16,
            num_heads=2,
            num_layers=1,
            self_attention=False,
        ),
        projection_head_config=ProjectionHeadConfig(
            out_dim=64,
            hidden_dim=32,
            bottleneck_dim=16,
        ),
        pqstem_enc_attn=("ge_attn", "attn"),
        pqstem_in_chans=32,
        pqstem_transdown_fps=(16, 12),
        pqstem_transdown_dims=(16, 32),
        pqstem_transdown_num_heads=(1, 2),
        pqstem_transdown_sa_depth=(1, 1),
        pqstem_transdown_k=(4, 4),
        sparse_visible_mask_ratio=0.5,
        sparse_masked_mask_ratio=0.5,
        geometric_halfspace_mask_ratio=0.5,
        random_mask_ratio=0.5,
        attn_num_layers=1,
        mask_probability=1.0,
        add_unmasked_global_cls=True,
        koleo_loss_weight=None,
        gradient_checkpointing=False,
    )
    model.setup()
    model.on_fit_start()
    output = model.training_step({"points": torch.randn(2, 32, 3)}, 0)

    assert torch.isfinite(output["loss"])
    assert output["patch_loss"] is not None


def test_pqstem_packed_mask_layout_uses_sparse_halfspace_random_paths():
    _skip_if_local_stem_deps_unavailable()
    (
        RandomPatchMasking,
        ProjectionHeadConfig,
        TransformerDecoderConfig,
        TransformerEncoderConfig,
        PQStemPackedFusedAttnBlockAsymDSD,
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
                fromlist=["TransformerDecoderConfig"],
            ).TransformerDecoderConfig,
            __import__(
                "asymdsd.layers",
                fromlist=["TransformerEncoderConfig"],
            ).TransformerEncoderConfig,
            __import__(
                "asymdsd.models.asymdsd_pqstem_fab_packed",
                fromlist=["PQStemPackedFusedAttnBlockAsymDSD"],
            ).PQStemPackedFusedAttnBlockAsymDSD,
        )
    )

    model = PQStemPackedFusedAttnBlockAsymDSD(
        max_epochs=1,
        steps_per_epoch=1,
        batch_size=2,
        mask_generator=RandomPatchMasking(mask_ratio=0.7, multi_mask=4),
        encoder_config=TransformerEncoderConfig(
            embed_dim=32,
            num_heads=2,
            num_layers=1,
        ),
        predictor_config=TransformerDecoderConfig(
            embed_dim=16,
            num_heads=2,
            num_layers=1,
            self_attention=False,
        ),
        projection_head_config=ProjectionHeadConfig(
            out_dim=64,
            hidden_dim=32,
            bottleneck_dim=16,
        ),
        pqstem_enc_attn=("ge_attn", "attn"),
        pqstem_in_chans=32,
        pqstem_transdown_fps=(16, 12),
        pqstem_transdown_dims=(16, 32),
        pqstem_transdown_num_heads=(1, 2),
        pqstem_transdown_sa_depth=(1, 1),
        pqstem_transdown_k=(4, 4),
        sparse_visible_mask_ratio=0.7,
        sparse_masked_mask_ratio=0.5,
        geometric_halfspace_mask_ratio=0.5,
        random_mask_ratio=0.7,
        gradient_checkpointing=False,
    )

    assert model._packed_mask_counts(10) == [7, 5, 5, 7]

    random_mask = model._generate_random_patch_mask(torch.randn(3, 10, 3), 7)
    assert random_mask.shape == (3, 10)
    assert random_mask.sum(dim=-1).tolist() == [7, 7, 7]


def test_pqstem_transup_head_exports_pqdt_keys():
    _skip_if_local_stem_deps_unavailable()
    PQStemTransUpHead = _import_or_skip(
        lambda: (
            __import__(
                "asymdsd.models.pq_transup",
                fromlist=["PQStemTransUpHead"],
            ).PQStemTransUpHead
        )
    )

    head = PQStemTransUpHead(
        embed_dim=16,
        num_seed=8,
        up_factors=(1, 2),
        n_knn=4,
    )
    state_dict = head.pqdt_up_layers_state_dict()

    assert state_dict
    assert all(key.startswith("up_layers.") for key in state_dict)
    assert "up_layers.0.mlp_1.mlp.0.weight" in state_dict
    assert head.output_sizes == (8, 8, 16)


def test_pqstem_packed_fab_tiny_training_step_with_transup():
    _skip_if_local_stem_deps_unavailable()
    (
        RandomPatchMasking,
        ProjectionHeadConfig,
        TransformerDecoderConfig,
        TransformerEncoderConfig,
        PQStemPackedFusedAttnBlockAsymDSD,
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
                fromlist=["TransformerDecoderConfig"],
            ).TransformerDecoderConfig,
            __import__(
                "asymdsd.layers",
                fromlist=["TransformerEncoderConfig"],
            ).TransformerEncoderConfig,
            __import__(
                "asymdsd.models.asymdsd_pqstem_fab_packed",
                fromlist=["PQStemPackedFusedAttnBlockAsymDSD"],
            ).PQStemPackedFusedAttnBlockAsymDSD,
        )
    )

    model = PQStemPackedFusedAttnBlockAsymDSD(
        max_epochs=1,
        steps_per_epoch=1,
        batch_size=2,
        mask_generator=RandomPatchMasking(mask_ratio=0.5, multi_mask=4),
        encoder_config=TransformerEncoderConfig(
            embed_dim=32,
            num_heads=2,
            num_layers=1,
        ),
        predictor_config=TransformerDecoderConfig(
            embed_dim=16,
            num_heads=2,
            num_layers=1,
            self_attention=False,
        ),
        projection_head_config=ProjectionHeadConfig(
            out_dim=64,
            hidden_dim=32,
            bottleneck_dim=16,
        ),
        pqstem_enc_attn=("ge_attn", "attn"),
        pqstem_in_chans=32,
        pqstem_transdown_fps=(16, 12),
        pqstem_transdown_dims=(16, 32),
        pqstem_transdown_num_heads=(1, 2),
        pqstem_transdown_sa_depth=(1, 1),
        pqstem_transdown_k=(4, 4),
        pqstem_enable_transup_reconstruction=True,
        pqstem_transup_num_seed=8,
        pqstem_transup_up_factors=(1, 2),
        pqstem_transup_n_knn=4,
        pqstem_transup_cd_weight=0.1,
        sparse_visible_mask_ratio=0.5,
        sparse_masked_mask_ratio=0.5,
        geometric_halfspace_mask_ratio=0.5,
        random_mask_ratio=0.5,
        attn_num_layers=1,
        mask_probability=1.0,
        koleo_loss_weight=None,
        gradient_checkpointing=False,
    )
    model.setup()
    model.on_fit_start()
    output = model.training_step({"points": torch.randn(2, 32, 3)}, 0)

    assert torch.isfinite(output["loss"])
    assert torch.isfinite(output["transup_cd_loss"])
    assert output["transup_reconstructions"][-1].shape == (2, 16, 3)


def test_pqstem_transup_validation_step_logs_cd_loss():
    _skip_if_local_stem_deps_unavailable()
    (
        RandomPatchMasking,
        ProjectionHeadConfig,
        TransformerDecoderConfig,
        TransformerEncoderConfig,
        PQStemPackedFusedAttnBlockAsymDSD,
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
                fromlist=["TransformerDecoderConfig"],
            ).TransformerDecoderConfig,
            __import__(
                "asymdsd.layers",
                fromlist=["TransformerEncoderConfig"],
            ).TransformerEncoderConfig,
            __import__(
                "asymdsd.models.asymdsd_pqstem_fab_packed",
                fromlist=["PQStemPackedFusedAttnBlockAsymDSD"],
            ).PQStemPackedFusedAttnBlockAsymDSD,
        )
    )

    model = PQStemPackedFusedAttnBlockAsymDSD(
        max_epochs=1,
        steps_per_epoch=1,
        batch_size=2,
        mask_generator=RandomPatchMasking(mask_ratio=0.5, multi_mask=4),
        encoder_config=TransformerEncoderConfig(
            embed_dim=32,
            num_heads=2,
            num_layers=1,
        ),
        predictor_config=TransformerDecoderConfig(
            embed_dim=16,
            num_heads=2,
            num_layers=1,
            self_attention=False,
        ),
        projection_head_config=ProjectionHeadConfig(
            out_dim=64,
            hidden_dim=32,
            bottleneck_dim=16,
        ),
        pqstem_enc_attn=("ge_attn", "attn"),
        pqstem_in_chans=32,
        pqstem_transdown_fps=(16, 12),
        pqstem_transdown_dims=(16, 32),
        pqstem_transdown_num_heads=(1, 2),
        pqstem_transdown_sa_depth=(1, 1),
        pqstem_transdown_k=(4, 4),
        pqstem_enable_transup_reconstruction=True,
        pqstem_transup_num_seed=8,
        pqstem_transup_up_factors=(1, 2),
        pqstem_transup_n_knn=4,
        sparse_visible_mask_ratio=0.5,
        sparse_masked_mask_ratio=0.5,
        geometric_halfspace_mask_ratio=0.5,
        random_mask_ratio=0.5,
        attn_num_layers=1,
        mask_probability=1.0,
        koleo_loss_weight=None,
        gradient_checkpointing=False,
    )
    model.setup()
    model.on_fit_start()

    logged = {}
    model.log = lambda name, value, **kwargs: logged.setdefault(name, value)
    output = model.validation_step({"points": torch.randn(2, 32, 3)}, 0)

    assert torch.isfinite(output["transup_cd_loss"])
    assert torch.isfinite(output["transup_last_cd2_loss"])
    assert output["transup_reconstructions"][-1].shape == (2, 16, 3)
    assert "val/transup_cd_loss" in logged
    assert "val/transup_last_cd2_loss" in logged


def test_pqstem_transup_cd_gradients_are_isolated():
    _skip_if_local_stem_deps_unavailable()
    PQStemPackedFusedAttnBlockAsymDSD = _import_or_skip(
        lambda: (
            __import__(
                "asymdsd.models.asymdsd_pqstem_fab_packed",
                fromlist=["PQStemPackedFusedAttnBlockAsymDSD"],
            ).PQStemPackedFusedAttnBlockAsymDSD
        )
    )

    model = PQStemPackedFusedAttnBlockAsymDSD(
        max_epochs=1,
        steps_per_epoch=1,
        batch_size=2,
        pqstem_enc_attn=("ge_attn", "attn"),
        pqstem_in_chans=32,
        pqstem_transdown_fps=(16, 12),
        pqstem_transdown_dims=(16, 32),
        pqstem_transdown_num_heads=(1, 2),
        pqstem_transdown_sa_depth=(1, 1),
        pqstem_transdown_k=(4, 4),
        pqstem_enable_transup_reconstruction=True,
        pqstem_transup_num_seed=8,
        pqstem_transup_up_factors=(1, 2),
        pqstem_transup_n_knn=4,
        gradient_checkpointing=False,
    )
    model.setup()
    model.on_fit_start()
    model.zero_grad(set_to_none=True)

    cd_loss, _ = model._transup_reconstruction_loss(torch.randn(2, 32, 3))
    cd_loss.backward()

    assert any(
        param.grad is not None and torch.any(param.grad != 0)
        for param in model.transup_head.up_layers.parameters()
    )
    assert all(param.grad is None for param in model.student.parameters())
    assert all(param.grad is None for param in model.teacher.parameters())


def test_pqstem_transup_export_loads_strictly_into_up_layers():
    _skip_if_local_stem_deps_unavailable()
    PQStemTransUpHead, UpLayer = _import_or_skip(
        lambda: (
            __import__(
                "asymdsd.models.pq_transup",
                fromlist=["PQStemTransUpHead"],
            ).PQStemTransUpHead,
            __import__(
                "asymdsd.models.pq_transup",
                fromlist=["UpLayer"],
            ).UpLayer,
        )
    )

    head = PQStemTransUpHead(
        embed_dim=16,
        num_seed=8,
        up_factors=(1, 2),
        n_knn=4,
    )
    exported = head.pqdt_up_layers_state_dict(cpu=False)
    stripped = {
        key.removeprefix("up_layers."): value for key, value in exported.items()
    }
    target = torch.nn.ModuleList(
        [
            UpLayer(dim=16, seed_dim=16, up_factor=1, i=0, n_knn=4),
            UpLayer(dim=16, seed_dim=16, up_factor=2, i=1, n_knn=4),
        ]
    )
    incompatible = target.load_state_dict(stripped, strict=True)

    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []
