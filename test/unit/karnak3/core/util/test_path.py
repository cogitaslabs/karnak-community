import unittest

import karnak3.core.util as ku


class TestStringMethods(unittest.TestCase):
    def test_path_concat(self):
        self.assertEqual(ku.path_concat(None, '/b/'), 'b')
        self.assertEqual(ku.path_concat('a', 'b'), 'a/b')
        self.assertEqual(ku.path_concat('/a', '/b'), '/a/b')
        self.assertEqual(ku.path_concat('a', None), 'a')
        self.assertEqual(ku.path_concat('/a/', '/b/'), '/a/b')
        self.assertEqual(ku.path_concat('/a/', None, '/b/'), '/a/b')
        self.assertEqual(ku.path_concat('/a/b/c', None, '/d/'), '/a/b/c/d')


# if __name__ == "__main__":
#     unittest.main()
