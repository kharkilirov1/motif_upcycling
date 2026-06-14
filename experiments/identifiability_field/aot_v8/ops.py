from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple
import torch

# ------------------------- D4: symmetries of a square -------------------------
D4_OPS = ["id", "rot90", "rot180", "rot270", "flip_h", "flip_v", "flip_diag", "flip_anti_diag"]
D4_OP2ID = {x: i for i, x in enumerate(D4_OPS)}
D4_ID2OP = {i: x for x, i in D4_OP2ID.items()}


def d4_transform(op_id: int, r: int, c: int, n: int = 3) -> Tuple[int, int]:
    if op_id == 0: return r, c
    if op_id == 1: return c, n - 1 - r
    if op_id == 2: return n - 1 - r, n - 1 - c
    if op_id == 3: return n - 1 - c, r
    if op_id == 4: return r, n - 1 - c
    if op_id == 5: return n - 1 - r, c
    if op_id == 6: return c, r
    if op_id == 7: return n - 1 - c, n - 1 - r
    raise ValueError(op_id)


def build_d4_table(n: int = 3) -> List[List[int]]:
    """table[new][state] = new ∘ state."""
    perms: Dict[Tuple[Tuple[int, int], ...], int] = {}
    coords = [(r, c) for r in range(n) for c in range(n)]
    for op in range(8):
        perms[tuple(d4_transform(op, r, c, n) for r, c in coords)] = op
    table = [[0] * 8 for _ in range(8)]
    for new in range(8):
        for state in range(8):
            composed = []
            for r, c in coords:
                r1, c1 = d4_transform(state, r, c, n)
                r2, c2 = d4_transform(new, r1, c1, n)
                composed.append((r2, c2))
            table[new][state] = perms[tuple(composed)]
    return table


D4_TABLE = build_d4_table()

# ------------------------- C8: cyclic group under addition -------------------------
C8_OPS = [f"add{i}" for i in range(8)]
C8_OP2ID = {x: i for i, x in enumerate(C8_OPS)}
C8_ID2OP = {i: x for x, i in C8_OP2ID.items()}


def build_cyclic_table(n: int = 8) -> List[List[int]]:
    """table[new][state] = (new + state) mod n."""
    return [[(new + state) % n for state in range(n)] for new in range(n)]


C8_TABLE = build_cyclic_table(8)

# ------------------------- Boolean unary monoid -------------------------
# Represent unary boolean functions by their outputs on (0, 1):
# id=(0,1), not=(1,0), false=(0,0), true=(1,1)
BOOL_OPS = ["keep", "invert", "false", "true"]
BOOL_FUNS = [(0, 1), (1, 0), (0, 0), (1, 1)]
BOOL_OP2ID = {x: i for i, x in enumerate(BOOL_OPS)}
BOOL_ID2OP = {i: x for x, i in BOOL_OP2ID.items()}


def build_bool_table() -> List[List[int]]:
    """table[new][state] = new ∘ state for unary boolean functions."""
    fun_to_id = {f: i for i, f in enumerate(BOOL_FUNS)}
    table = [[0] * 4 for _ in range(4)]
    for new_id, new_fun in enumerate(BOOL_FUNS):
        for state_id, state_fun in enumerate(BOOL_FUNS):
            # (new ∘ state)(x) = new(state(x))
            out0 = new_fun[state_fun[0]]
            out1 = new_fun[state_fun[1]]
            table[new_id][state_id] = fun_to_id[(out0, out1)]
    return table


BOOL_TABLE = build_bool_table()

DOMAINS = ["d4", "c8", "bool"]
DOMAIN2ID = {"d4": 1, "c8": 2, "bool": 3}   # token-label kind ids; 0 is none
ID2DOMAIN = {v: k for k, v in DOMAIN2ID.items()}
DOMAIN_CLASS = {"d4": 0, "c8": 1, "bool": 2} # final task-kind ids
DOMAIN_SIZES = {"d4": 8, "c8": 8, "bool": 4}
DOMAIN_TABLES = {"d4": D4_TABLE, "c8": C8_TABLE, "bool": BOOL_TABLE}
DOMAIN_OPS = {"d4": D4_OPS, "c8": C8_OPS, "bool": BOOL_OPS}
FLAT_OFFSETS = {"d4": 0, "c8": 8, "bool": 16}
FLAT_OUT = 20
MAX_OPS = 8


def compose_sequence(table: Sequence[Sequence[int]], seq: Sequence[int], identity: int = 0) -> int:
    state = identity
    for op in seq:
        state = int(table[int(op)][state])
    return state


def check_associative(table: Sequence[Sequence[int]]) -> bool:
    n = len(table)
    for a in range(n):
        for b in range(n):
            for c in range(n):
                left = table[a][table[b][c]]
                right = table[table[a][b]][c]
                if left != right:
                    return False
    return True


def soft_compose_step(state_dist: torch.Tensor, op_dist: torch.Tensor, table: Sequence[Sequence[int]]) -> torch.Tensor:
    """Soft composition for small finite monoids/groups.

    state_dist: [B, N], distribution over current composed state.
    op_dist:    [B, N], distribution over new operator.
    table[o][s] gives o ∘ s.
    returns: [B, N]
    """
    bsz, n = state_dist.shape
    out = state_dist.new_zeros(bsz, n)
    # N is tiny (4 or 8); the explicit loop is clearer and stable on CPU.
    for op in range(n):
        op_p = op_dist[:, op]
        if torch.is_grad_enabled() or bool((op_p.detach().abs() > 0).any()):
            for state in range(n):
                out_state = int(table[op][state])
                out[:, out_state] = out[:, out_state] + op_p * state_dist[:, state]
    return out


def one_hot(indices: torch.Tensor, n: int) -> torch.Tensor:
    return torch.nn.functional.one_hot(indices.long(), n).float()


def soft_compose_from_onehots(seq: Sequence[int], table: Sequence[Sequence[int]]) -> int:
    state = torch.zeros(1, len(table))
    state[0, 0] = 1.0
    for op in seq:
        op_dist = torch.zeros(1, len(table))
        op_dist[0, int(op)] = 1.0
        state = soft_compose_step(state, op_dist, table)
    return int(state.argmax(-1).item())
