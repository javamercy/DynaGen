import unittest

import numpy as np

from dynagen.candidates.validation import validate_generated_code
from dynagen.execution import sandbox as sandbox_module


_NUMBA_TSP_SOLVER = """
import numpy as np
from numba import njit

@njit
def _identity_tour(n):
    tour = np.empty(n, dtype=np.int64)
    for idx in range(n):
        tour[idx] = idx
    return tour

def solve_tsp(distance_matrix):
    return _identity_tour(distance_matrix.shape[0])
"""


class NumbaIntegrationTests(unittest.TestCase):
    def test_validation_accepts_numba_imports(self) -> None:
        result = validate_generated_code(_NUMBA_TSP_SOLVER)

        self.assertTrue(result.valid, result.error)

    def test_validation_rejects_numba_cache(self) -> None:
        result = validate_generated_code("""
from numba import njit

@njit(cache=True)
def _identity_tour(n):
    return n

def solve_tsp(distance_matrix):
    return list(range(len(distance_matrix)))
""")

        self.assertFalse(result.valid)
        self.assertEqual(result.error, "Numba caching is not allowed")

    def test_sandbox_imports_numba_or_errors_clearly(self) -> None:
        if "numba" not in sandbox_module.ALLOWED_MODULES:
            with self.assertRaisesRegex(ImportError, "Optional dependency numba is not installed"):
                sandbox_module.load_tsp_solver(_NUMBA_TSP_SOLVER)
            return

        solver = sandbox_module.load_tsp_solver(_NUMBA_TSP_SOLVER)
        tour = solver(np.zeros((4, 4), dtype=float))

        np.testing.assert_array_equal(tour, np.array([0, 1, 2, 3]))


if __name__ == "__main__":
    unittest.main()
