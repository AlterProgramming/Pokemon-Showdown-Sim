from __future__ import annotations

import unittest

import numpy as np

from visualize_entity_embeddings import cosine_neighbors, normalize_points, pca_2d


class VisualizeEntityEmbeddingsTests(unittest.TestCase):
    def test_pca_2d_preserves_row_count(self) -> None:
        matrix = np.asarray(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        projected = pca_2d(matrix)
        self.assertEqual(projected.shape, (3, 2))

    def test_normalize_points_scales_into_canvas(self) -> None:
        points = np.asarray([[0.0, 0.0], [2.0, 4.0]], dtype=np.float32)
        scaled = normalize_points(points, width=100.0, height=50.0)
        self.assertTrue(np.all(scaled[:, 0] >= 0.0))
        self.assertTrue(np.all(scaled[:, 1] >= 0.0))

    def test_cosine_neighbors_omits_self(self) -> None:
        matrix = np.asarray(
            [
                [1.0, 0.0],
                [0.9, 0.1],
                [0.0, 1.0],
            ],
            dtype=np.float32,
        )
        neighbors = cosine_neighbors(matrix, top_k=1)
        self.assertEqual(neighbors[0], [1])
        self.assertEqual(neighbors[1], [0])


if __name__ == "__main__":
    unittest.main()
