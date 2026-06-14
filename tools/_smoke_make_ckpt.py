"""Create a tiny LeWorldModel checkpoint for smoke-testing eval_goals.py."""
import sys
from pathlib import Path

import torch
from torch.optim import SGD

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from wally.models.lewm import LeWorldModel  # noqa: E402
from wally.training.checkpoint import save_checkpoint  # noqa: E402

model_cfg = {
    "vit_variant": "vit_tiny_patch16_224",
    "embed_dim": 192,
    "depth": 4,
    "num_heads": 4,
    "mlp_ratio": 4.0,
    "dropout": 0.0,
    "action_dim": 25,
    "pretrained": False,
}
model = LeWorldModel(**model_cfg)
opt = SGD(model.parameters(), lr=1e-4)
out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("checkpoints/_smoke_dummy.pt")
out.parent.mkdir(parents=True, exist_ok=True)
save_checkpoint(
    out,
    model,
    opt,
    global_step=1234,
    config={"model": model_cfg},
)
print("wrote", out)
