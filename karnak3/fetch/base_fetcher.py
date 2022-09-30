from __future__ import annotations

from abc import abstractmethod
from typing import Dict, List, Optional, Union, Any, Generator
import datetime
import time
import pandas as pd
import threading
import copy
import orjson
import pytz
import gc

import karnak3.core.util as ku
import karnak3.core.log as kl
import karnak3.core.profiling as kp


def json_serial(obj):  # FIXME move to somewhere else, or use orjson
    """JSON serializer for objects not serializable by default json code"""
    if isinstance(obj, (datetime.datetime, datetime.date)):
        return obj.isoformat()
    raise TypeError("Type %s not serializable" % type(obj))


class FetcherQueueItem:
    def __init__(self, keys: List[str],
                 key_property: str,
                 table: str,
                 extractor: str,
                 current_retries: int = 0,
                 creation_ts: Optional[datetime.datetime] = None,
                 update_dt: Optional[datetime.datetime] = None,
                 cohort: Optional[str] = None,
                 priority: Optional[int] = None,
                 fetch_errors: Optional[List[str]] = None,
                 handle: str = None):
        now = datetime.datetime.now(tz=pytz.utc)
        self.keys: List[str] = keys
        self.key_property: str = key_property
        self.table: str = table
        self.extractor: str = extractor
        self.creation_ts: datetime.datetime = creation_ts if creation_ts is not None else now
        self.update_dt: datetime.datetime = update_dt if update_dt is not None else now
        self.cohort: Optional[str] = cohort
        self.priority: Optional[int] = priority
        self.current_retries: int = current_retries
        self.fetch_errors: Optional[List[str]] = fetch_errors if fetch_errors is not None else []
        self.handle: str = handle

    def add_error(self, fetch_error) -> FetcherQueueItem:
        ret = self.copy()
        ret.current_retries += 1
        ret.fetch_errors.append(fetch_error)
        ret.update_dt = datetime.datetime.now(tz=pytz.utc)
        return ret

    def copy(self) -> FetcherQueueItem:
        return copy.copy(self)

    def restart(self, extractor: str = None) -> FetcherQueueItem:
        ret = self.copy()
        ret.fetch_errors = []
        ret.current_retries = 0
        ret.update_dt = datetime.datetime.now(tz=pytz.utc)
        ret.extractor = self.extractor if extractor is None else extractor
        ret.handle = None
        return ret

    def set_priority(self, priority) -> FetcherQueueItem:
        c = self.copy()
        if priority is not None:
            c.priority = priority
        return c

    def set_extractor(self, extractor, reset_errors=False) -> FetcherQueueItem:
        c = self.copy()
        if extractor is not None:
            c.extractor = extractor
        if reset_errors:
            c.current_retries = 0
            # c.fetch_errors = []
        return c

    def to_string(self):
        filtered_dict = self.__dict__.copy()
        del filtered_dict['handle']
        # s = orjson.dumps(filtered_dict, default=json_serial).decode()
        s = orjson.dumps(filtered_dict, default=json_serial).decode()
        return s

    @classmethod
    def from_string(cls, s: str, handle=None):
        d = orjson.loads(s)
        try:
            f = FetcherQueueItem(keys=d.get('keys'),
                                 key_property=d.get('key_property'),
                                 table=d.get('table'),
                                 extractor=d.get('extractor'),
                                 creation_ts=datetime.datetime.fromisoformat(d.get('creation_ts')),
                                 update_dt=datetime.datetime.fromisoformat(d.get('update_dt')),
                                 cohort=d.get('cohort'),
                                 priority=d.get('priority'),
                                 current_retries=d.get('current_retries'),
                                 # fetch_errors=d.get('fetch_errors'),
                                 handle=handle)
        except KeyError:
            msg = s.replace("\n", "")
            kl.error(f'invalid message: {msg}')
            return None
        return f


class FetcherResult:
    def __init__(self, queue_item: FetcherQueueItem,
                 data: Union[None, dict, list, str],
                 rows: int,
                 capture_ts: datetime.datetime,
                 elapsed: datetime.timedelta,
                 is_success: bool = True,
                 can_retry: bool = False,
                 compression: str = None,
                 error_type: str = None,
                 error_message: str = None,
                 response_edition: Optional[List[str]] = None,
                 handle: str = None):
        self.queue_item = queue_item
        self.data: Union[None, dict, list, str] = data
        self.rows: int = rows
        self.capture_ts: datetime.datetime = capture_ts
        self.elapsed: datetime.timedelta = elapsed
        self.is_success = is_success
        self.can_retry = can_retry
        self.compression = compression
        self.error_type = error_type
        self.error_message = error_message
        self.response_edition: Optional[List[str]] = response_edition
        self.handle = handle

    def elapsed_str(self):
        return str(self.elapsed)

    def flat_dict(self, include_handle: bool = False) -> dict:
        flat_dict = self.__dict__.copy()
        if self.queue_item is not None:
            flat_dict.update(self.queue_item.__dict__)
        if not include_handle:
            del flat_dict['handle']
        else:
            flat_dict['handle'] = self.handle
        del flat_dict['queue_item']
        return flat_dict

    def to_string(self):
        filtered_dict = self.__dict__.copy()
        del filtered_dict['handle']
        filtered_dict['elapsed'] = self.elapsed.total_seconds()
        filtered_dict['queue_item'] = self.queue_item.to_string()
        s = orjson.dumps(filtered_dict).decode()
        return s

    @classmethod
    def from_string(cls, s: str, handle=None):
        try:
            d = orjson.loads(s)
            elapsed_raw = d.get('elapsed')
            elapsed = None if elapsed_raw is None else datetime.timedelta(seconds=elapsed_raw)
            queue_item = FetcherQueueItem.from_string(d['queue_item'])

            fr = FetcherResult(queue_item=queue_item,
                               data=d.get('data'),
                               capture_ts=d.get('capture_ts'),
                               elapsed=elapsed,
                               rows=d.get('rows'),
                               is_success=d.get('is_success'),
                               can_retry=d.get('can_retry'),
                               compression=d.get('compression'),
                               error_type=d.get('error_type'),
                               error_message=d.get('error_message'),
                               response_edition=d.get('response_edition'),
                               handle=handle)
            return fr
        except KeyError:
            msg = s.replace("\n", " ")
            kl.error(f'invalid message: {msg}')
            return None


class KarnakFetcherThreadContext:
    def __init__(self, context_vars: dict = None):
        # general context dict for those who do not wish to create a subclass
        self.context_vars: dict =  context_vars if context_vars is not None else {}


class KarnakFetcher:
    def __init__(self, name: str,
                 tables: List[str],
                 environment: str,
                 extractors: Optional[List[str]] = None,
                 max_priority: Optional[int] = None):
        self.name: str = name
        self.tables: List[str] = tables
        self.environment: str = environment
        self.extractors: Optional[List[str]] = extractors if extractors is not None else ['default']
        self.max_priority: Optional[int] = max_priority

    #
    # general
    #

    def priorities(self) -> List[Optional[int]]:
        return [None] if self.max_priority is None else list(range(self.max_priority, 0, -1)) + [None]

    @abstractmethod
    def fetcher_state(self, queue_sizes: Optional[Dict[str, int]] = None) -> (str, int):
        """
        Returns de state of the fetcher
        ;param queue_sizes: dict with precalculated (queue name, queue size)
        :return: (str) one of the values: 'idle' (all worker queues empty), 'working' (worker queues with data),
                'consolidating' (only results queue not empty)
                (int) smallest priority of queue with pending items
        """
        pass

    @staticmethod
    def compress_data(data: Union[None, list, dict],
                      compression: str = 'brotli',
                      compression_level=None) -> Optional[str]:
        if data is None:
            return None
        elif compression is None:
            return None
        else:
            nice_json_bytes = orjson.dumps(data)
            if compression == 'brotli':
                quality = 9 if compression_level is None else compression_level
                compressed_data = ku.compress_base64(nice_json_bytes, compression='brotli',
                                                     quality=quality)
            elif compression == 'zlib':
                compressed_data = ku.compress_base64(nice_json_bytes, compression='zlib',
                                                     level=compression_level)
            else:
                compressed_data = nice_json_bytes.decode()

            return compressed_data

    #
    # kickoff
    #

    @abstractmethod
    def keys_to_fetch(self, table: str,
                      max_keys: Optional[int] = None,
                      add_keys: Optional[List[str]] = None,
                      method: Optional[str] = None,
                      scope: Optional[str] = None,
                      priority: Optional[int] = None,
                      **args) -> List[FetcherQueueItem]:
        """
        Returns list of keys to fetch
        :param table: table to fetch
        :param max_keys: max number of keys to fetch
        :param add_keys: add this keys to fetch
        :param method: method to find keys to fetch
        :param scope: scope to limit keys fetched
        :param args: additional arguments needed for derived classes
        :param priority: priority of queue to put items
        :param dry_run: if True, dos not put items in queue
        :return: list of items to be fetched
        """
        return []

    def kickoff_ready(self, empty_priority:  Optional[int] = None) -> (bool, str):
        state, working_priority = self.fetcher_state()
        if state != 'working':
            return True, state
        elif empty_priority is None:
            return False, state
        else:
            return (working_priority > empty_priority), state

    def set_initial_extractor(self, items: List[FetcherQueueItem]):
        return items

    @abstractmethod
    def populate_worker_queue(self, items: List[FetcherQueueItem],
                              extractor: str,
                              priority: Optional[int],
                              threads: int = 1):
        pass

    def kickoff(self, table: str,
                max_keys: Optional[int] = None,
                add_keys: Optional[List[str]] = None,
                method: Optional[str] = None,
                scope: Optional[str] = None,
                priority: Optional[int] = None,
                if_empty: bool = False,
                wait_empty: bool = False,
                empty_priority: Optional[int] = None,
                extractors: List[str] = None,
                threads: int = 1,
                dry_run: bool = False,
                **args) -> bool:

        _extractors = extractors if extractors is not None else self.extractors

        # test if its ready to kickoff
        if if_empty:
            kickoff_ready, state = self.kickoff_ready(empty_priority)
            if not kickoff_ready:
                kl.info(f'cannot kickoff {self.name} table {table}: current state is {state}.')
                return False
        elif wait_empty:
            wait_time_seconds = 60
            while True:
                kickoff_ready, state = self.kickoff_ready(empty_priority)
                if kickoff_ready:
                    break
                kl.info(f'waiting {wait_time_seconds}s for kickoff {self.name} table {table}:'
                        f' current state is {state}.')
                time.sleep(wait_time_seconds)

        # keys and initial strategies
        items = self.keys_to_fetch(table=table, max_keys=max_keys, add_keys=add_keys,
                                   method=method, scope=scope, **args)

        if items is None or len(items) == 0:
            kl.info(f'cannot kickoff {self.name} table {table}: nothing to fetch.')
            return False

        # set priority, cohort, creation time
        if priority is not None:
            items = [x.set_priority(priority) for x in items]

        # set initial extractor
        if len(self.extractors) > 0:
            items = self.set_initial_extractor(items)

        for extractor in _extractors:
            extractor_items = [x for x in items if x.extractor == extractor]
            kl.debug(f'populating extractor {extractor} with {len(extractor_items)} items.')
            if dry_run:
                kl.info(f'dry run enabled: will not put {len(extractor_items)} items in queue.')
            else:
                self.populate_worker_queue(extractor_items, extractor=extractor, priority=priority,
                                           threads=threads)

        kl.debug(f'kickoff completed for {self.name} table {table}.')

    #
    # worker
    #

    @abstractmethod
    def pop_work_queue_item(self, extractor: str,
                            priority: Optional[int],
                            context: KarnakFetcherThreadContext,
                            wait: bool) -> Optional[FetcherQueueItem]:
        pass

    @abstractmethod
    def pop_best_work_queue_item(self, extractor: str,
                                 context: KarnakFetcherThreadContext) \
            -> Optional[FetcherQueueItem]:
        pass

    #
    # consolidator
    #

    @abstractmethod
    def pop_result_items(self, max_items: int,
                         threads: int = 1) -> List[FetcherResult]:
        pass

    @abstractmethod
    def consolidate(self, max_queue_items_per_file: int = 120_000,
                    max_rows_per_file: int = 2_000_000,
                    threads: int = 1,
                    **args):
        pass

    def results_df(self, fetched_data: List[FetcherResult]) -> pd.DataFrame:
        flat_data = [x.flat_dict(include_handle=True) for x in fetched_data]
        return pd.DataFrame(flat_data)

    @abstractmethod
    def save_consolidation(self, prepared_df: pd.DataFrame, table: str,
                           time_slice_id: str, time_slice_ref: Any,
                           current_file: int, n_files: int, **args):
        pass

    @abstractmethod
    def prepare_row(self, result_row: pd.Series, decoded_item: dict) -> Optional[dict]:
        pass

    def time_slice_ref(self, df: pd.DataFrame) -> Any:
        """Identifier for time slice. by default, the latest timestamp"""
        return pd.to_datetime(df['capture_ts'].max())

    def time_slicing(self, df: pd.DataFrame) -> Generator[(pd.DataFrame, str, Any)]:
        """Generator for time slicing. default is by capture date.

        Returns the dataframe slice, a string identifier, and an arbitrary object that
            will be used by save function
        """
        dt_series = df['capture_ts'].str.slice(stop=10)
        dt_set = dt_series.unique()
        for dt_str in dt_set:
            dt_slice = df[dt_series == dt_str]
            slice_obj = self.time_slice_ref(dt_slice)
            yield dt_slice, dt_str, slice_obj

    def rows_slicing(self, df: pd.DataFrame, max_rows_per_file: int) \
            -> Generator[(pd.DataFrame, int, int)]:
        n_rows = df['rows'].sum()
        n_files = -(-n_rows // max_rows_per_file)  # rounds up
        rows_accumulator = []
        file_count = 0
        kl.debug(f'slicing {n_rows} rows from {len(df)} items into {n_files} files...')

        def file_slice() -> pd.DataFrame:
            nonlocal file_count, rows_accumulator
            file_count += 1
            next_slice_data = rows_accumulator[:max_rows_per_file]
            f_slice = pd.DataFrame(next_slice_data)
            rows_accumulator = rows_accumulator[max_rows_per_file:]
            return f_slice

        for index, result_row in df.iterrows():
            decoded_data_str = ku.decompress_str_base64(result_row['data'],
                                                        result_row['compression'])
            decoded_data_list = orjson.loads(decoded_data_str) if decoded_data_str is not None \
                else []
            data_rows = [self.prepare_row(result_row, decoded_item)
                         for decoded_item in decoded_data_list]
            data_rows = [i for i in data_rows if i is not None]  # remove failed rows
            rows_accumulator.extend(data_rows)
            while len(rows_accumulator) >= max_rows_per_file:
                yield file_slice(), file_count, n_files

        while len(rows_accumulator) > 0:
            yield file_slice(), file_count, n_files

    def custom_max_rows_per_file(self, table: str) -> Optional[int]:
        return None

    def _effective_max_rows_per_file(self, table: str, max_rows_per_file: int) -> int:
        custom_mrpf = self.custom_max_rows_per_file(table)
        effective_mrpf = max_rows_per_file
        if custom_mrpf:
            effective_mrpf = min(custom_mrpf, max_rows_per_file)
            kl.trace(f'effective max_rows_per_file for table {table}: {effective_mrpf}')
        return effective_mrpf

    def prepare_consolidation(self, fetched_df: pd.DataFrame, max_rows_per_file: int,
                              threads: int = 1, **args):
        if fetched_df is None or len(fetched_df) == 0:
            kl.info('empty dataframe, nothing to save.')

        kl.info(f'saving consolidated data for {len(fetched_df)} rows...')
        kprof = kp.KProfiler()
        kprof.log_mem('memory usage')

        # slice by table
        tables = set(fetched_df['table'].unique())
        for table in tables:
            _max_rows_per_file = self._effective_max_rows_per_file(table, max_rows_per_file)
            table_slice = fetched_df.loc[fetched_df['table'] == table]
            kl.debug(f'preparing data for table {table} ({len(table_slice)} items)...')
            self.prepare_table_consolidation(table_slice)

            # slice by time
            for (time_slice_df, time_slice_id, time_slice_ref) in self.time_slicing(table_slice):

                # slice into files and prepare dataframe
                for (prepared_file_df, current_file, n_files) \
                        in self.rows_slicing(time_slice_df, _max_rows_per_file):

                    n_rows = len(prepared_file_df)
                    n_items = len(prepared_file_df['handle'].unique())
                    kl.debug(f"prepared slice #{current_file} from table {table} with {n_rows} rows from "
                             f"{n_items} items")
                    if len(prepared_file_df) == 0:
                        kl.warn('saving file with 0 valid rows')
                    self.save_consolidation(prepared_file_df, table, time_slice_id=time_slice_id,
                                            time_slice_ref=time_slice_ref,
                                            current_file=current_file, n_files=n_files, **args)
                    self.clean_slice_consolidation(prepared_file_df, table, threads=threads)

                    kprof.log_mem('memory usage before gc')
                    del prepared_file_df
                    gc.collect()
                    kprof.log_mem('memory usage after gc')

    def prepare_table_consolidation(self, table_slice_df: pd.DataFrame) -> Any:
        pass

    @abstractmethod
    def clean_slice_consolidation(self, prepared_file_df: pd.DataFrame,
                                  table: str, threads: int = 1):
        pass


class KarnakFetcherWorker:
    def __init__(self, fetcher: KarnakFetcher, extractor: str, n_threads: int = 1,
                 retries: int = 1,
                 loop_pause_seconds: int = 180,
                 stop_after_queue_items: Optional[int] = None,
                 stop_after_minutes: Optional[int] = None):
        self.fetcher: KarnakFetcher = fetcher
        self.extractor = extractor
        self.n_threads = n_threads
        self.loop_pause_seconds = loop_pause_seconds
        self.retries = retries
        self.state = 'working'
        self.stop_after_queue_items: Optional[int] = stop_after_queue_items
        self.stop_after_minutes: Optional[int] = stop_after_minutes
        self.processed_queue_items_cnt: int = 0
        self.work_started_at: Optional[datetime.datetime] = None
        self.work_stop_at: Optional[datetime.datetime] = None
        self.idle_counter = 0
        # self.state_check_lock = threading.RLock()

    @ku.synchronized
    def inc_processed_queue_items_counter(self):
        self.processed_queue_items_cnt += 1

    def set_stop_time(self):
        if self.work_started_at is None and self.stop_after_minutes is not None:
            self.work_started_at = datetime.datetime.now()
            self.work_stop_at = datetime.datetime.now() \
                                + datetime.timedelta(minutes=self.stop_after_minutes)

    def should_keep_walking(self) -> bool:
        keep_walking_items = self.processed_queue_items_cnt < self.stop_after_queue_items \
            if self.stop_after_queue_items is not None else True
        keep_walking_time = datetime.datetime.now() < self.work_stop_at \
            if self.work_stop_at is not None else True

        keep_walking = keep_walking_items and keep_walking_time
        if not keep_walking_items:
            kl.info(f'stoping worker: items counter reached {self.processed_queue_items_cnt}')
        if not keep_walking_time:
            kl.info(f'stoping worker: time run time reached {self.stop_after_minutes} minute(s)')
        return keep_walking

    def throttle_request(self):
        pass

    @abstractmethod
    def pop_work_queue_item(self, priority: Optional[int],
                            context: KarnakFetcherThreadContext, wait: bool) \
            -> Optional[FetcherQueueItem]:
        pass

    @abstractmethod
    def pop_best_work_queue_item(self,
                                 context: KarnakFetcherThreadContext) \
            -> Optional[FetcherQueueItem]:
        pass

    @abstractmethod
    def return_queue_item(self, item: FetcherQueueItem, context: KarnakFetcherThreadContext):
        pass

    @abstractmethod
    def push_queue_item(self, item: FetcherQueueItem, context: KarnakFetcherThreadContext):
        pass

    @abstractmethod
    def refresh_queue_item(self, item: FetcherQueueItem,
                           context: KarnakFetcherThreadContext,
                           new_extractor: str = None,
                           new_priority: int = None):
        pass

    @abstractmethod
    def complete_queue_item(self, item: FetcherResult, context: KarnakFetcherThreadContext):
        pass

    @abstractmethod
    def fetch_item(self, item: FetcherQueueItem, context: KarnakFetcherThreadContext) \
            -> FetcherResult:
        pass

    @abstractmethod
    def decide_failure_action(self, result: FetcherResult) -> (str, str):
        """Returns:
                ('abort', None): do not try again.
                ('ignore', None): return message to queue.
                ('retry', None): retry in same strategy.
                ('retry', 'xpto'): move to extractor xpto.
                ('restart', 'xpto'): save in current extractor, and also reset errors and create
                    new task in new extractor
        """
        return 'abort', None

    @classmethod
    def new_thread_context(cls) -> KarnakFetcherThreadContext:
        ctx = KarnakFetcherThreadContext()
        return ctx

    def process_item(self, item: FetcherQueueItem, thread_context: KarnakFetcherThreadContext):
        result = self.fetch_item(item, thread_context)

        if result.is_success:
            # successful ones: move to results
            self.complete_queue_item(result, thread_context)
            kl.debug(f"success fetching {result.queue_item.table} keys "
                     f" {result.queue_item.keys} in {result.elapsed_str()}, "
                     f"attempt {result.queue_item.current_retries}")
        else:  # failure
            action, new_extractor = self.decide_failure_action(result)
            if action == 'abort':
                self.complete_queue_item(result, thread_context)
            elif action == 'ignore':
                self.return_queue_item(item, thread_context)
            elif action == 'restart':
                restart_item = result.queue_item.restart(extractor=new_extractor)
                self.push_queue_item(restart_item, thread_context)
                self.complete_queue_item(result, thread_context)
            elif action == 'retry':
                retry_item = result.queue_item
                retry_item = retry_item.add_error(result.error_type)
                # if new_extractor is not None and new_extractor != self.extractor:
                #     item.strategy = new_extractor
                self.refresh_queue_item(retry_item, new_extractor)
            else:
                assert False  # Oh, no! It can't be! All is lost!

    def check_worker_state(self) -> str:
        fetcher_state, working_priority = self.fetcher.fetcher_state()
        if fetcher_state == 'working':
            self.set_state('working')
        else:
            self.set_state('idle', force=True)
            self.idle_counter = 0
        return self.state


    @ku.synchronized
    def reset_idle_counter(self):
        if self.idle_counter != 0:
            kl.trace(f'idle counter reset')
        self.idle_counter = 0

    @ku.synchronized
    def set_state(self, state: str, force: bool = False):
        idle_threshold = 3
        if state == 'idle':
            self.idle_counter += 1
            kl.trace(f'idle counter: {self.idle_counter}')
            if force or self.idle_counter >= idle_threshold:
                self.state = 'idle'
                kl.trace(f'worker set to idle')
        else:
            self.state = state
            self.reset_idle_counter()

    def fetcher_thread_loop(self, thread_num: int):
        context = self.new_thread_context()
        while self.state == 'working' and self.should_keep_walking():
            self.throttle_request()
            # self.state_check_lock.acquire()
            item = self.pop_best_work_queue_item(context=context)
            if item is None:
                kl.trace(f'thread {thread_num}: no item available in queue')
                self.set_state('idle')
            else:
                kl.trace(f'thread {thread_num}: read item')
                self.process_item(item, context)
                self.inc_processed_queue_items_counter()
                self.reset_idle_counter()
        kl.trace(f'thread {thread_num}: finished')

    def loop_pause(self):
        kl.trace(f'loop pause: {self.loop_pause_seconds} s')
        time.sleep(self.loop_pause_seconds)

    def worker_loop(self):
        while self.should_keep_walking():
            self.work()
            self.loop_pause()
        kl.info('worker loop finished.')

    def work(self):
        self.set_stop_time()
        self.check_worker_state()
        if self.state != 'working':
            kl.warn(f'Nothing to do: worker in state {self.state}')
            return

        threads = []
        for i in range(self.n_threads):
            t = threading.Thread(target=self.fetcher_thread_loop, args=(i,))
            t.start()
            threads.append(t)

        # wait for queue to be processed
        for t in threads:
            t.join()

        kl.info(f'worker for {self.extractor} finished: fetcher state {self.state}')

    def pack_result(self, item: FetcherQueueItem,
                    data: Union[list, dict, str, None],
                    rows: int,
                    capture_ts: datetime.datetime,
                    elapsed: datetime.timedelta,
                    compression: str = None,
                    compression_level=None,
                    is_success: Optional[bool] = None,
                    can_retry: Optional[bool] = None,
                    error_type: str = None,
                    error_message: str = None,
                    response_edition: str = None) -> FetcherResult:
        # is_success and can_retry are inferred if not set by caller
        _is_success = is_success if is_success is not None else (data is not None
                                                                 and error_type is None)
        _can_retry = can_retry if can_retry is not None else not _is_success
        _error_type = error_type if _is_success or error_type is not None else 'fetch-error'
        _error_message = error_message if _is_success or error_message is not None \
            else 'fetch error'
        compressed_data = data

        try:
            if data is not None and compression is not None:
                compressed_data = self.fetcher.compress_data(data, compression=compression,
                                                             compression_level=compression_level)

        except Exception as e:
            kl.exception(f'compression error: exception for {item.table}, keys {item.keys}', e)
            _can_retry = False
            _error_type = 'compression-error'
            _error_message = str(e)

        fetcher_result = FetcherResult(item, data=compressed_data,
                                       capture_ts=capture_ts,
                                       rows=rows,
                                       elapsed=elapsed, is_success=_is_success,
                                       can_retry=_can_retry, compression=compression,
                                       error_type=_error_type, error_message=_error_message,
                                       response_edition=response_edition)
        return fetcher_result


