import unittest
import pytz
from datetime import timedelta

import karnak3.core.util as ku
import karnak3.core.time_window as ktw
from karnak3.core.util import kdatetime, kdate


class TestTimeWindow(unittest.TestCase):
    def test_validate_time_window(self):

        ts1 = ku.kts(2021, 1, 2, 3, 4, 5)
        ts2 = ku.kts(2021, 1, 3, 4, 5, 6)
        dt1 = kdate(2021, 1, 2)

        t1, msg = ktw.validate_time_window(window_start=ts1,
                                           window_end=ts2,
                                           window_date=dt1)
        self.assertFalse(t1)

        t2, msg = ktw.validate_time_window(window_start=ts1,
                                           window_end=ts2,
                                           days=2)
        self.assertFalse(t2)

        t3a, msg = ktw.validate_time_window(window_start=ts2,
                                            window_end=ts1)
        self.assertFalse(t3a)

        t3b, msg = ktw.validate_time_window(window_start=ts1, window_end=ts2)
        self.assertEqual((t3b, msg), (True, ''))

        t4, msg = ktw.validate_time_window(window_start=ts1, days=5)
        self.assertEqual((t4, msg), (True, ''))

    #
    # time window slices
    #
    def test_decode_time_window_slices(self):

        tz = pytz.utc
        ts1 = ku.kts(2022, 1, 1, 0, 0, 0, tz=tz)
        ts1b = ku.kts(2022, 1, 1, 12, 0, 0, tz=tz)
        ts2 = ku.kts(2022, 1, 2, 0, 0, 0, tz=tz)
        ts3 = ku.kts(2022, 1, 3, 0, 0, 0, tz=tz)
        ts3b = ku.kts(2022, 1, 3, 12, 0, 0, tz=tz)
        dt1_1 = kdate(2022, 1, 1)
        dt1_2 = kdate(2022, 1, 2)
        dt1_3 = kdate(2022, 1, 3)
        dt1_4 = kdate(2022, 1, 4)
        dt1_15 = kdate(2022, 1, 15)
        dt1_16 = kdate(2022, 1, 16)
        dt1_17 = kdate(2022, 1, 17)
        dt2_16 = kdate(2022, 2, 16)
        today_dt = ku.today(tz=tz)
        today_ts = ku.dt_to_ts(today_dt, tz=tz)
        tomorrow_dt = today_dt + timedelta(days=1)
        tomorrow_ts = today_ts + timedelta(days=1)
        yesterday_dt = today_dt - timedelta(days=1)
        yesterday_ts = today_ts - timedelta(days=1)
        current_year = today_dt.year
        current_month_start = kdate(today_dt.year, today_dt.month, 1)
        next_month_start = (current_month_start + timedelta(days=32)).replace(day=1)
        previous_month_start = (current_month_start - timedelta(days=1)).replace(day=1)

        # window_start and window_end
        t1a = ktw.decode_time_window_slices(window_start=ts1, window_end=ts2)
        self.assertEqual([(ts1, ts2)], t1a)

        t1b = ktw.decode_time_window_slices(window_start=ts1, window_end=ts2, time_type='date')
        self.assertEqual([(dt1_1, dt1_2)], t1b)

        t1c = ktw.decode_time_window_slices(window_start=dt1_1, window_end=dt1_2, time_type='date')
        self.assertEqual([(dt1_1, dt1_2)], t1c)

        tx = ktw.decode_time_window_slices(window_start=ts1, window_end=ts3b, time_type='date')
        self.assertEqual([(dt1_1, dt1_4)], tx)

        tx = ktw.decode_time_window_slices(window_start=ts1, window_end=ts3b, time_type='date',
                                           frequency='day')
        self.assertEqual([(dt1_1, dt1_2), (dt1_2, dt1_3), (dt1_3, dt1_4)], tx)

        # test days
        tx = ktw.decode_time_window_slices(window_start=ts1, days=2, time_type='date',
                                           frequency='day')
        self.assertEqual([(dt1_1, dt1_2), (dt1_2, dt1_3)], tx)

        tx = ktw.decode_time_window_slices(window_end=dt1_3, days=2, time_type='date')
        self.assertEqual([(dt1_1, dt1_3)], tx)

        tx = ktw.decode_time_window_slices(window_end=ts3, days=2, time_type='date')
        self.assertEqual([(dt1_1, dt1_3)], tx)

        tx = ktw.decode_time_window_slices(window_end=ts3b, days=2, time_type='date')
        # in this case, rounding up get 3 days. we believe this is the right thing.
        self.assertEqual([(dt1_1, dt1_4)], tx)

        tx = ktw.decode_time_window_slices(window_end=ts3b, days=2, time_type='datetime')
        self.assertEqual([(ts1b, ts3b)], tx)

        # test year
        tx = ktw.decode_time_window_slices(year=2021, time_type='date')
        self.assertEqual([(kdate(2021, 1, 1), kdate(2022, 1, 1))], tx)

        # test window_date
        tx = ktw.decode_time_window_slices(window_date=dt1_2, time_type='datetime')
        self.assertEqual([(ts2, ts3)], tx)

        # test weekdays
        tx = ktw.decode_time_window_slices(window_start=dt1_1, window_end=dt1_17,
                                           weekdays=['sun', 'tue'], time_type='date',
                                           frequency='day')
        txs = ktw.time_window_slices_start(tx)
        self.assertEqual([dt1_2, dt1_4, kdate(2022, 1, 9), kdate(2022, 1, 11),
                          dt1_16], txs)

        # test last_weekday
        tx = ktw.decode_time_window_slices(window_start=dt1_1, window_end=dt1_16,
                                           last_weekday='sun', time_type='date',
                                           frequency='day')
        txs = ktw.time_window_slices_start(tx)
        self.assertEqual([kdate(2022, 1, 9)], txs)

        tx = ktw.decode_time_window_slices(window_start=dt1_1, window_end=dt1_17,
                                           last_weekday='sun', time_type='date',
                                           frequency='day')
        txs = ktw.time_window_slices_start(tx)
        self.assertEqual([dt1_16], txs)

        #
        # test auto_window
        #

        # today
        tx = ktw.decode_time_window_slices(auto_window='today', time_type='date')
        self.assertEqual([(today_dt, tomorrow_dt)], tx)

        tx = ktw.decode_time_window_slices(auto_window='today', time_type='datetime')
        self.assertEqual([(today_ts, tomorrow_ts)], tx)

        tx = ktw.decode_time_window_slices(auto_window='today', days=2, time_type='date')
        self.assertEqual([(yesterday_dt, tomorrow_dt)], tx)

        # yesterday with days
        tx = ktw.decode_time_window_slices(auto_window='yesterday', time_type='date')
        self.assertEqual([(yesterday_dt, today_dt)], tx)

        tx = ktw.decode_time_window_slices(auto_window='yesterday', time_type='datetime')
        self.assertEqual([(yesterday_ts, today_ts)], tx)

        tx = ktw.decode_time_window_slices(auto_window='yesterday', days=2, time_type='date')
        self.assertEqual([(yesterday_dt - timedelta(days=1), today_dt)], tx)

        # full
        tx = ktw.decode_time_window_slices(auto_window='full', time_type='datetime')
        self.assertEqual([], tx)

        # last-year
        tx = ktw.decode_time_window_slices(auto_window='last-year', time_type='date')
        self.assertEqual([(kdate(current_year-1, 1, 1), kdate(current_year, 1, 1))], tx)

        # ytd
        tx = ktw.decode_time_window_slices(auto_window='ytd', time_type='date')
        self.assertEqual([(kdate(current_year, 1, 1), today_dt)], tx)

        # last-month
        tx = ktw.decode_time_window_slices(auto_window='last-month', time_type='date')
        self.assertEqual([(previous_month_start, current_month_start)], tx)

        # mtd
        tx = ktw.decode_time_window_slices(auto_window='mtd', time_type='date')
        self.assertEqual([(current_month_start, today_dt)], tx)

        # filename
        tx = ktw.decode_time_window_slices(auto_window='filename', time_type='date',
                                           filename='xpto-file-20220102-ypt.xxx')
        self.assertEqual([(dt1_2, dt1_2)], tx)

        tx = ktw.decode_time_window_slices(auto_window='filename', time_type='datetime',
                                           filename='xpto-file-20220101:120000-ypt.xxx')
        self.assertEqual([(ts1b, ts1b)], tx)

        #
        # test frequency
        #

        # hour
        tx = ktw.decode_time_window_slices(auto_window='yesterday', time_type='datetime',
                                           frequency='hour')
        self.assertEqual([(yesterday_ts, yesterday_ts + timedelta(hours=1)),
                          (yesterday_ts + timedelta(hours=1), yesterday_ts + timedelta(hours=2))],
                         tx[:2])
        self.assertEqual([(today_ts - timedelta(hours=2), today_ts - timedelta(hours=1)),
                          (today_ts - timedelta(hours=1), today_ts)], tx[-2:])

        # TODO: day

        # TODO: month

        # TODO: year

        # TODO: 30min

        # TODO: 15 min

        # TODO: minute


