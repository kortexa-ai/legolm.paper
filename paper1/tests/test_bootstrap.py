import tempfile
from pathlib import Path
import unittest

from legolm_p01.bootstrap import download_verified
from legolm_p01.provenance import file_hash


class ExactInputs(unittest.TestCase):
    def test_existing_wrong_input_is_rejected_without_overwriting(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/"input.bin"
            p.write_bytes(b"existing-user-input")
            with self.assertRaises(ValueError):
                download_verified("https://unused.invalid",p,{"sha256":"0"*64,"bytes":len(p.read_bytes())})
            self.assertEqual(p.read_bytes(),b"existing-user-input")
            download_verified("https://unused.invalid",p,{"sha256":file_hash(p),"bytes":len(p.read_bytes())})


if __name__ == "__main__":
    unittest.main()
