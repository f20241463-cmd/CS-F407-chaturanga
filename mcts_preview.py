"""
mcts_preview.py

Worked example of wiring real MCTS into the actual chaturanga
environment, in the same spirit as Student 1's integration_preview.py.

Run this from the repo root AFTER copying the mcts/ folder in (see
README in this bundle for exactly where it goes). Requires the
`chaturanga` package from Student 1's part of the repo to already be
importable.

    python mcts_preview.py
"""

from chaturanga.game import GameState
from chaturanga import encoding
from chaturanga.stub_network import random_stub_network, zero_stub_network

from mcts.search import run_mcts


def main():
    state = GameState()

    print("Running MCTS with the RANDOM stub network (50 simulations)...")
    visit_counts, move = run_mcts(
        root_state=state,
        network_fn=random_stub_network,
        num_simulations=50,
        c_puct=1.5,
        state_to_tensor_fn=encoding.state_to_tensor,
        legal_action_mask_fn=encoding.legal_action_mask,
        action_to_move_fn=encoding.action_to_move,
    )
    print(f"  Chosen move: {move}")
    print(f"  Total root visits: {int(visit_counts.sum())}")
    print(f"  Number of root children explored: {(visit_counts > 0).sum()}")

    print("\nRunning MCTS with the ZERO stub network (uniform policy, value=0)...")
    print("  (useful sanity check: with no signal at all, visits should spread")
    print("  roughly evenly across legal first moves)")
    visit_counts, move = run_mcts(
        root_state=state,
        network_fn=zero_stub_network,
        num_simulations=50,
        c_puct=1.5,
        state_to_tensor_fn=encoding.state_to_tensor,
        legal_action_mask_fn=encoding.legal_action_mask,
        action_to_move_fn=encoding.action_to_move,
    )
    print(f"  Chosen move: {move}")
    nonzero_visits = visit_counts[visit_counts > 0]
    print(f"  Visit counts across explored moves: {nonzero_visits}")


if __name__ == "__main__":
    main()
