"""CPU checks for feature graph residuals; no dataset or checkpoints required."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from ultralytics.nn.modules.head import AttributeFeatureGraph, v10MDetect


def main():
    torch.set_num_threads(2)
    for operator in ("gca", "gcn", "gat", "graphsage", "gin", "local"):
        torch.manual_seed(7)
        graph = AttributeFeatureGraph(8, 3, 2, None, operator)
        features = torch.randn(2, 8, 3, 4)
        logits = torch.randn(2, 6, 3, 4)
        assert torch.equal(graph(features, logits), logits)
        optimizer = torch.optim.SGD(graph.parameters(), lr=0.1)
        for _ in range(3):
            optimizer.zero_grad()
            result = graph(features, logits).reshape(2, 3, 2, 3, 4)
            loss = torch.nn.functional.cross_entropy(result.permute(0, 2, 1, 3, 4),
                                                     torch.zeros(2, 3, 3, 4, dtype=torch.long))
            loss.backward()
            optimizer.step()
        assert graph.project.weight.grad.abs().sum() > 0
        assert not torch.equal(graph(features, logits), logits)
        if operator != "local":
            assert graph.message.weight.grad.abs().sum() > 0
            original = graph(features, logits).detach()
            graph.adjacency.zero_()
            assert not torch.allclose(original, graph(features, logits), atol=1e-8, rtol=0)
        graph.enabled = False
        assert torch.equal(graph(features, logits), logits)
        print(operator, "identity, learning, matrix influence and bypass OK")

    # The gain is an explicit amplitude control: with identical parameters,
    # changing gain from 1 to 4 should scale only the residual correction.
    torch.manual_seed(13)
    unit = AttributeFeatureGraph(8, 3, 2, None, "gca", gain=1.0)
    with torch.no_grad():
        unit.fusion[-1].weight.normal_()
        unit.fusion[-1].bias.normal_()
    amplified = AttributeFeatureGraph(8, 3, 2, None, "gca", gain=4.0)
    amplified.load_state_dict(unit.state_dict())
    unit_output = unit(features, logits)
    amplified_output = amplified(features, logits)
    assert torch.allclose(
        amplified_output - logits,
        4.0 * (unit_output - logits),
        atol=1e-6,
        rtol=1e-5,
    )
    print("feature_gain: residual amplitude scales exactly")

    base = v10MDetect(nc=2, na=3, nal=2, params=[False, None, None, False, None], ch=(16, 32, 64))
    model = v10MDetect(nc=2, na=3, nal=2, params=[False, None, 'feature_gca_cross', False, None], ch=(16, 32, 64))
    gained_model = v10MDetect(
        nc=2,
        na=3,
        nal=2,
        params=[False, None, 'feature_gca_cross', False, None, 4.0],
        ch=(16, 32, 64),
    )
    assert gained_model.feature_gain == 4.0
    assert all(graph.feature_gain == 4.0 for graph in gained_model.gat_head)
    incompatible = model.load_state_dict(base.state_dict(), strict=False)
    assert not incompatible.unexpected_keys
    assert all('gat_head' in key for key in incompatible.missing_keys)
    x = [torch.randn(2, c, s, s) for c, s in ((16, 8), (32, 4), (64, 2))]
    base.eval()
    model.eval()
    # Compare the actual training outputs without BatchNorm running updates.
    base.training = model.training = True
    expected, actual = base([v.clone() for v in x]), model([v.clone() for v in x])
    for branch in ('one2many', 'one2one'):
        assert all(torch.equal(a, b) for a, b in zip(expected[branch], actual[branch]))
    sum(v.square().mean() for branch in actual.values() for v in branch).backward()
    assert model.gat_head[0].fusion[-1].weight.grad.abs().sum() > 0
    assert model.one2one_gat_head[0].fusion[-1].weight.grad.abs().sum() > 0
    print('v10MDetect: original keys preserved, both branches equivalent at initialization and receive gradients')


if __name__ == '__main__':
    main()
