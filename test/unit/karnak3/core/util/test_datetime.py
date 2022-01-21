import unittest
import pytz

import karnak3.core.util as ku
from karnak3.core.util import kdatetime, kdate


class TestTimeWindow(unittest.TestCase):

    ku.set_default_tz(pytz.utc)

    def test_parse_filename_date(self):
        tx = ku.parse_filename_date('xpto-file-20220102-ypt.xxx')
        self.assertEqual(kdate(2022, 1, 2), tx)

    def test_parse_filename_timestamp(self):
        tx = ku.parse_filename_datetime('xpto-file-20220102-ypt.xxx')
        self.assertEqual(kdatetime(2022, 1, 2, tzinfo=pytz.utc), tx)

        tx = ku.parse_filename_datetime('xpto-file-20220102:100000-ypt.xxx')
        self.assertEqual(kdatetime(2022, 1, 2, 10, 0, 0, tzinfo=pytz.utc), tx)
