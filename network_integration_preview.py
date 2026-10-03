"""
network_integration_preview.py

First real test of your MCTS against Student 2's actual ChaturangaNet
(untrained, random weights). Same spirit as mcts_preview.py, which
tested against the stub -- this one tests against the real thing.

Only run this after merging main into nn-model locally, so both
mcts/ and network/ exist together.

Run from the repo root:
    python network_integration_preview.py
"""

from chaturanga.game import GameState
from chaturanga import encoding

from network.model import ChaturangaNet
from network.agent import make_network_fn

from mcts.search import run_mcts


def main():
    print("Building an untrained ChaturangaNet (random weights)...")
    net = ChaturangaNet()  # defaults: 4 res blocks, 64 filters, matches team's target
    network_fn = make_network_fn(net)

    state = GameState()

    print("\nSanity check: does network_fn alone return the right shapes?")
    tensor = encoding.state_to_tensor(state)
    policy, value = network_fn(tensor)
    print(f"  policy shape: {policy.shape}  (expect (4096,))")
    print(f"  value: {value:.4f}  (expect a single float in [-1, 1])")
    assert policy.shape == (4096,), (
        "policy shape mismatch -- stop here, report to Student 2"
    )
    assert -1.0 <= value <= 1.0, "value out of range -- stop here, report to Student 2"
    print("  OK")

    print("\nRunning MCTS with the real (untrained) network, 50 simulations...")
    visit_counts, move = run_mcts(
        root_state=state,
        network_fn=network_fn,
        num_simulations=50,
        c_puct=1.5,
        state_to_tensor_fn=encoding.state_to_tensor,
        legal_action_mask_fn=encoding.legal_action_mask,
        action_to_move_fn=encoding.action_to_move,
    )
    print(f"  Chosen move: {move}")
    print(f"  Total root visits: {int(visit_counts.sum())}")
    print(f"  Number of root children explored: {(visit_counts > 0).sum()}")

    print("\nRunning it again on the SAME network instance, to confirm determinism:")
    print("  (an untrained real network gives a fixed prediction per position,")
    print("  unlike the stub which drew fresh randomness every call -- visit")
    print("  counts concentrating on fewer moves than the stub did is expected,")
    print("  not a bug)")
    visit_counts2, move2 = run_mcts(
        root_state=state,
        network_fn=network_fn,
        num_simulations=50,
        c_puct=1.5,
        state_to_tensor_fn=encoding.state_to_tensor,
        legal_action_mask_fn=encoding.legal_action_mask,
        action_to_move_fn=encoding.action_to_move,
    )
    print(f"  Chosen move: {move2}  (should match the first run exactly)")

    print("\nIf nothing crashed and both runs agree: the plumbing between")
    print("your MCTS and his network is confirmed working end to end.")


if __name__ == "__main__":
    main()
