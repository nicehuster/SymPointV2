import unittest

import torch

from modules.pointops.functions import pointops


def make_points(count):
    x = torch.linspace(0.0, 1.0, count, device="cuda", dtype=torch.float32)
    zeros = torch.zeros_like(x)
    xyz = torch.stack((x, zeros, zeros), dim=1).contiguous()
    offset = torch.cuda.IntTensor([count])
    return xyz, offset


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class KNNQueryCapacityTest(unittest.TestCase):
    def test_supports_256_neighbors_without_invalid_indexes(self):
        xyz, offset = make_points(256)
        new_xyz = xyz[:1].contiguous()
        new_offset = torch.cuda.IntTensor([1])

        indexes, distances = pointops.knnquery(
            256, xyz, new_xyz, offset, new_offset
        )
        torch.cuda.synchronize()

        self.assertEqual(tuple(indexes.shape), (1, 256))
        self.assertEqual(tuple(distances.shape), (1, 256))
        self.assertGreaterEqual(indexes.min().item(), 0)
        self.assertLess(indexes.max().item(), xyz.shape[0])

        features = torch.arange(
            256, device="cuda", dtype=torch.float32
        ).reshape(256, 1).contiguous()
        output = pointops.interpolation(
            xyz, new_xyz, features, offset, new_offset, k=256
        )
        torch.cuda.synchronize()

        self.assertEqual(tuple(output.shape), (1, 1))
        self.assertTrue(torch.isfinite(output).all().item())

    def test_rejects_neighbor_count_above_capacity(self):
        xyz, offset = make_points(257)
        new_xyz = xyz[:1].contiguous()
        new_offset = torch.cuda.IntTensor([1])

        with self.assertRaisesRegex(
            RuntimeError,
            r"knnquery nsample must be between 1 and 256; got 257",
        ):
            pointops.knnquery(257, xyz, new_xyz, offset, new_offset)


if __name__ == "__main__":
    unittest.main()
