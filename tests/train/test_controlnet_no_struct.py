"""Physical struct removal preserves the trained model with empty struct keys."""
import copy
import pytest
import torch
from tests.train.test_compact_controlnet import fixture, forward
from streamvoxel.models.compact_controlnet import CompactStreamingControlNet
from streamvoxel.train.train_controlnet import control_state, load_control_state


def test_pruned_checkpoint_matches_empty_struct_forward_and_gradients():
    original, group = fixture()
    with torch.no_grad():
        for projection in original.zero:
            projection.weight.normal_(std=.01)
    group['struct_labels'] = group['struct_labels'][:0]
    group['struct_sizes'] = group['struct_sizes'][:0]
    group['prefix_positions'] = group['prefix_positions'][:group['l2_count']]
    pruned = CompactStreamingControlNet(copy.deepcopy(original.backbone), control_depth=3,
        control_hidden_dim=24, checkpointing=False, biome_halo=4, biome_patch=2, struct_classes=0)
    load_control_state(pruned, control_state(original))
    assert pruned.config()['struct_classes'] == 0
    assert pruned.type_embed.num_embeddings == 3
    assert not any(n.startswith(('struct_embed.', 'struct_size.')) for n in pruned.state_dict())
    full, reduced = forward(original, group), forward(pruned, group)
    torch.testing.assert_close(full, reduced, rtol=0, atol=0)
    full.square().mean().backward(); reduced.square().mean().backward()
    reference = dict(original.named_parameters())
    for name, value in pruned.named_parameters():
        if not value.requires_grad:
            assert value.grad is None
            continue
        grad = reference[name].grad
        if name == 'type_embed.weight':
            grad = grad[:3]
        assert value.grad is not None and torch.isfinite(value.grad).all(), name
        torch.testing.assert_close(value.grad, grad, rtol=0, atol=0)
    # Reject position leakage rather than quietly misaligning the prefix.
    group['prefix_positions'] = torch.randn(2, 3)
    with pytest.raises(ValueError, match='Struct-free'):
        pruned.encode_group(group)


def test_struct_free_no_l2_has_empty_prefix_and_finite_forward():
    original, group = fixture()
    pruned = CompactStreamingControlNet(copy.deepcopy(original.backbone), control_depth=3,
        control_hidden_dim=24, checkpointing=False, biome_halo=4, biome_patch=2, struct_classes=0)
    load_control_state(pruned, control_state(original))
    group.update(l2_blocks=group['l2_blocks'][:0], l2_count=0,
        struct_labels=group['struct_labels'][:0], struct_sizes=group['struct_sizes'][:0],
        prefix_positions=group['prefix_positions'][:0])
    assert pruned.encode_group(group)['prefix'].shape == (0, 24)
    assert torch.isfinite(forward(pruned, group)).all()
