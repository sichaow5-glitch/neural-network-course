"""Small mathematical/architecture checks; run with python test_models.py."""

import json

import torch
from torch import nn
from torchvision.models import resnet18

from models import VARIANTS, build_model, profile_model


def main() -> None:
    torch.set_num_threads(4)
    torch.manual_seed(7)
    x = torch.randn(2, 3, 64, 64)
    baseline = build_model("baseline").eval()
    reference = resnet18(weights=None, num_classes=43).eval()
    reference.load_state_dict(baseline.state_dict(), strict=True)
    with torch.inference_mode():
        torch.testing.assert_close(baseline(x), reference(x), rtol=0, atol=0)
    assert sum(p.numel() for p in baseline.parameters()) == 11_198_571

    profiles = {}
    for variant in VARIANTS:
        model = build_model(variant).train()
        # Mixed module modes test exact restoration rather than just root mode.
        model.layer2.eval()
        modes = [m.training for m in model.modules()]
        bn_before = {
            key: value.clone() for key, value in model.state_dict().items()
            if "running_" in key or "num_batches_tracked" in key
        }
        profiles[variant] = profile_model(model)
        assert modes == [m.training for m in model.modules()]
        for key, before in bn_before.items():
            torch.testing.assert_close(before, model.state_dict()[key], rtol=0, atol=0)
        assert profiles[variant]["output_size"] == [1, 43]
        out = model(x)
        assert out.shape == (2, 43)
        nn.functional.cross_entropy(out, torch.tensor([0, 42])).backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        if "depthwise" in variant:
            assert sum(isinstance(m, nn.Conv2d) and m.groups > 1 for m in model.modules()) == 16
            assert all(len(list(block.conv1.children())) == 2 for layer in
                       [model.layer1, model.layer2, model.layer3, model.layer4] for block in layer)

    assert profiles["depthwise"]["parameters"] < profiles["baseline"]["parameters"]
    assert profiles["depthwise"]["macs"] < profiles["baseline"]["macs"]
    assert profiles["se"]["parameters"] - profiles["baseline"]["parameters"] == 87_040
    assert profiles["depthwise_se"]["parameters"] - profiles["depthwise"]["parameters"] == 87_040
    print(json.dumps({"status": "passed", "torchvision_baseline_exact_match": True,
                      "profile_restores_modes_and_bn": True, "profiles": profiles}, indent=2))


if __name__ == "__main__":
    main()
