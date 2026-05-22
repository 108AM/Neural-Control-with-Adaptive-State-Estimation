import numpy as np

def _gramian_size(W) -> int:
    size: int = W.shape[0]
    assert len(W.shape) == 2 and size == W.shape[1], "Gramian must be a square matrix"
    return W.shape[0]

def get_rank(W, tol: float = 1e-10) -> int:
    _ = _gramian_size(W)

    return int(
        np.linalg.matrix_rank(W, tol=tol)
    )

def is_controllable(W_c, tol: float = 1e-10) -> bool:
    rank: int = get_rank(W_c, tol=tol)
    return rank == W_c.shape[0]

def is_observable(W_o, tol: float = 1e-10) -> bool:
    rank: int = get_rank(W_o, tol=tol)
    return rank == W_o.shape[0]

def controllability_condition(W_c) -> float:
    _ = _gramian_size(W_c)
    return np.linalg.cond(W_c)

def observability_condition(W_o) -> float:
    _ = _gramian_size(W_o)
    return np.linalg.cond(W_o)

# T = T if T is not None else self.state_dim
# Wc = self.controllability_gramian(T)
# Wo = self.observability_gramian(T)
# return {
#     "controllable": int(np.linalg.matrix_rank(Wc, tol=tol)) == self.state_dim,
#     "observable": int(np.linalg.matrix_rank(Wo, tol=tol)) == self.state_dim,
#     "controllability_rank": int(np.linalg.matrix_rank(Wc, tol=tol)),
#     "observability_rank": int(np.linalg.matrix_rank(Wo, tol=tol)),
#     "controllability_condition": float(np.linalg.cond(Wc)),
#     "observability_condition": float(np.linalg.cond(Wo)),
#     "T": T,
# }
