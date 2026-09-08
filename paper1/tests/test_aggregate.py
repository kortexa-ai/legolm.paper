import unittest

from legolm_p01.aggregate import paired_seeds


class PairedSeeds(unittest.TestCase):
    def test_regularization_uses_shared_seeds_and_reports_subset(self):
        rows = lambda seeds, kind="bridge": [{"recipe":{"seed":s,"kind":kind}} for s in seeds]
        seeds,left,right = paired_seeds(rows([42,1042,2042]), rows([42,1042,2042,3042,4042]))
        self.assertEqual(seeds,[42,1042,2042])
        self.assertEqual(len(left),3)
        self.assertEqual(len(right),3)
        seeds,left,right = paired_seeds(rows([42,1042,2042]), rows([42],"none"))
        self.assertEqual(len(left),3)
        self.assertEqual(len(right),1)
        with self.assertRaises(ValueError):
            paired_seeds(rows([42]), rows([2042]))


if __name__ == "__main__":
    unittest.main()
