from __future__ import annotations

import sys

from pyathena.util import synchronized

import karnak.util.log as kl
import karnak.util.aws.sqs as ksqs
import karnak.util.profiling as kp

from abc import abstractmethod, ABC
from typing import Dict, List, Optional, Union, Generator, Any
import datetime
import time
import pandas as pd
import threading
import copy
import orjson
import base64
import brotli
import zlib
import gc
import pytz


def json_serial(obj):
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

    def flat_dict(self) -> dict:
        flat_dict = self.__dict__.copy()
        if self.queue_item is not None:
            flat_dict.update(self.queue_item.__dict__)
        del flat_dict['handle']
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
    pass


class KarnakSqsFetcherThreadContext(KarnakFetcherThreadContext):
    def __init__(self):
        self.sqs_client = ksqs.get_client()
        super().__init__()


def compress_base64(b: bytes, compression: str, **args) -> Optional[str]:
    if b is None:
        return None
    if compression in ['zlib', 'gz', 'gzip', 'brotli']:
        if compression == 'brotli':
            bytes_enc = brotli.compress(b, **args)
        elif compression in ['zlib', 'gz', 'gzip']:
            bytes_enc = zlib.compress(b, **args)
        else:
            assert False
        enc = base64.b64encode(bytes_enc).decode('ascii')
        return enc
    else:
        return b.decode()


def decompress_str_base64(s: str, compression: str) -> Optional[str]:
    if s is None:
        return None
    if compression in ['zlib', 'gz', 'gzip', 'brotli']:
        b = base64.b64decode(s)
        if compression == 'brotli':
            decompressed = brotli.decompress(b)
        elif compression in ['zlib', 'gz', 'gzip']:
            decompressed = zlib.decompress(b)
        else:
            assert False
        return decompressed.decode()
    else:
        return s


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
        return [None] if self.max_priority is None else list(range(1, self.max_priority + 1)) + [None]

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
    def compress_data(data: Union[None, list, dict], compression: str = 'brotli', compression_level=None)\
            -> Optional[str]:
        if data is None:
            return None
        elif compression is None:
            return None
        else:
            nice_json_bytes = orjson.dumps(data)
            if compression == 'brotli':
                quality = 9 if compression_level is None else compression_level
                compressed_data = compress_base64(nice_json_bytes, compression='brotli', quality=quality)
            elif compression == 'zlib':
                compressed_data = compress_base64(nice_json_bytes, compression='zlib', level=compression_level)
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
    def populate_worker_queue(self, items: List[FetcherQueueItem], extractor: str, priority: Optional[int]):
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
            self.populate_worker_queue(extractor_items, extractor=extractor, priority=priority)

        kl.debug(f'kickoff completed for {self.name} table {table}.')

    #
    # worker
    #

    @abstractmethod
    def pop_work_queue_item(self, extractor: str, priority: Optional[int],
                            context: KarnakFetcherThreadContext, wait: bool) \
            -> Optional[FetcherQueueItem]:
        pass

    @abstractmethod
    def pop_best_work_queue_item(self, extractor: str,
                                 context: KarnakFetcherThreadContext) -> Optional[FetcherQueueItem]:
        pass

    #
    # consolidator
    #

    @abstractmethod
    def pop_result_items(self, max_items) -> List[FetcherResult]:
        pass

    @abstractmethod
    def consolidate(self, max_queue_items_per_file: int = 120_000, max_rows_per_file: int = 2_000_000, **args):
        pass

    @abstractmethod
    def data_row(self, result: FetcherResult, data_item: Union[dict, list, str, None]) -> dict:
        pass

    def results_df(self, fetched_data: List[FetcherResult]) -> pd.DataFrame:
        flat_data = [x.flat_dict() for x in fetched_data]
        return pd.DataFrame(flat_data)

    def data_to_df(self, fetched_data: list) -> pd.DataFrame:
        fetched_df = pd.DataFrame(columns=['table', 'raw'])
        # mem_size_mb = int(sys.getsizeof(fetched_data) / (1024 * 1024))
        # kl.trace(f'full memory size: {mem_size_mb} MB')
        for table in self.tables:
            kl.trace(f'converting table {table} to dataframe...')
            split_results = [x for x in fetched_data if x.queue_item.table == table]
            # mem_size_table_mb = int(sys.getsizeof(split_results) / (1024 * 1024))
            kl.trace(f'{table} has {len(split_results)} items')
            split_rows = []
            for result in split_results:
                data_encoded = result.data
                data_decoded_str = decompress_str_base64(data_encoded, compression=result.compression)
                data = orjson.loads(data_decoded_str) if data_decoded_str is not None else []

                # multiple elements per fetched_data row
                new_rows = [self.data_row(result, data_item) for data_item in data]
                split_rows.extend(new_rows)

            if len(split_rows) > 0:
                df = pd.DataFrame(split_rows)
                fetched_df = fetched_df.append(df, ignore_index=True)

        return fetched_df


class KarnakSqsFetcher(KarnakFetcher):
    def __init__(self, name: str,
                 tables: List[str],
                 environment: str,
                 extractors: Optional[List[str]] = None,
                 max_priority: Optional[int] = None,
                 empty_work_queue_recheck_seconds: int = 300):
        super().__init__(name, tables, environment, extractors, max_priority)
        self.empty_queue_control = {}
        self.default_sqs_client = ksqs.get_client()
        self.empty_work_queue_recheck_seconds = empty_work_queue_recheck_seconds

    #
    # queues
    #

    @abstractmethod
    def results_queue_name(self) -> str:
        """Returns the name of the results queue."""
        pass

    @abstractmethod
    def worker_queue_name(self, extractor: str, priority: Optional[int]) -> str:
        """Returns the name of the worker queue."""
        pass

    def worker_queue_names(self, extractor=None) -> List[str]:
        priorities = self.priorities()
        _extractors = [extractor] if extractor is not None else self.extractors
        ql = [self.worker_queue_name(ext, p) for ext in _extractors for p in priorities]
        return ql

    def fetcher_state(self, queue_sizes: Optional[Dict[str, int]] = None) -> (str, int):
        if queue_sizes is None:
            queue_sizes = self.queue_sizes()
        qs_results = queue_sizes[self.results_queue_name()]
        qs_workers = sum([queue_sizes[qn] for qn in self.worker_queue_names()])
        working_priority = None
        if self.max_priority is not None and qs_workers > 0:
            for p in range(1, self.max_priority + 1):
                q_names = [self.worker_queue_name(ext, p) for ext in self.extractors]
                cnt = sum([queue_sizes[qn] for qn in q_names])
                if cnt > 0:
                    working_priority = p
                    break

        if qs_results + qs_workers == 0:
            return 'idle', working_priority
        elif qs_workers == 0:
            return 'consolidating', working_priority
        else:
            return 'working', working_priority

    def queue_sizes(self, sqs_client=None) -> Dict[str, int]:
        """Returns approximate message count for all queues."""
        kl.trace(f'getting queue sizes')
        _sqs_client = sqs_client if sqs_client is not None else self.default_sqs_client
        qs = {}
        queue_names = self.worker_queue_names() + [self.results_queue_name()]
        for q in queue_names:
            attr = ksqs.queue_attributes(q, sqs_client=_sqs_client)
            available = int(attr['ApproximateNumberOfMessages'])
            in_flight = int(attr['ApproximateNumberOfMessagesNotVisible'])
            delayed = int(attr['ApproximateNumberOfMessagesDelayed'])
            qs[q] = available + in_flight + delayed
        return qs

    #
    # kickoff
    #

    def populate_worker_queue(self, items: List[FetcherQueueItem], extractor: str, priority: Optional[int]):
        contents = [i.to_string() for i in items]
        ksqs.send_messages(self.worker_queue_name(extractor=extractor, priority=priority), contents)

    #
    # worker
    #

    def create_thread_context(self) -> KarnakSqsFetcherThreadContext:
        ctx = KarnakSqsFetcherThreadContext()
        return ctx

    @synchronized
    def set_empty_queue(self, queue_name: str):
        self.empty_queue_control[queue_name] = datetime.datetime.now(tz=pytz.utc)

    @synchronized
    def is_empty_queue(self, queue_name: str,) -> bool:
        eqc = self.empty_queue_control.get(queue_name)
        if eqc is None:
            return False
        now = datetime.datetime.now(tz=pytz.utc)
        if now - eqc >= datetime.timedelta(seconds=self.empty_work_queue_recheck_seconds):
            del self.empty_queue_control[queue_name]
            return False
        return True

    def pop_work_queue_item(self, extractor: str, priority: Optional[int],
                            context: KarnakSqsFetcherThreadContext, wait: bool) \
            -> Optional[FetcherQueueItem]:
        queue_name = self.worker_queue_name(extractor, priority=priority)
        sqs_client = context.sqs_client
        wait_seconds = 20 if wait else 0
        items = ksqs.receive_messages(queue_name=queue_name, max_messages=1, wait_seconds=wait_seconds,
                                      sqs_client=sqs_client)
        if items is None or len(items) == 0:
            self.set_empty_queue(queue_name)
            return None
        else:
            assert len(items) == 1
            handle = list(items.keys())[0]
            content_str = items[handle]
            ret = FetcherQueueItem.from_string(content_str, handle=handle)
            return ret

    def pop_best_work_queue_item(self, extractor: str,
                                 context: KarnakSqsFetcherThreadContext) -> Optional[FetcherQueueItem]:
        priorities = self.priorities()
        for retry in [0, 1]:  # two rounds of attempts
            for p in priorities:
                queue_name = self.worker_queue_name(extractor, priority=p)
                if retry or not self.is_empty_queue(queue_name):  # only checks empty in first round.
                    # only wait in retry round.
                    wait = retry > 0
                    item = self.pop_work_queue_item(extractor, p, context, wait=wait)
                    if item is not None:
                        return item

    #
    # consolidator
    #

    def pop_result_items(self, max_items) -> List[FetcherResult]:
        items = ksqs.receive_messages(queue_name=self.results_queue_name(),
                                      max_messages=max_items, wait_seconds=20)
        ret = [FetcherResult.from_string(items[handle], handle=handle) for handle in items if items is not None]
        return ret

    def consolidate(self, max_queue_items_per_file: int = 120_000, max_rows_per_file: int = 2_000_000, **args):
        kl.info(f'consolidate {self.name}: start.')
        kl.debug(f'max_queue_items_per_file: {max_queue_items_per_file}, max_rows_per_file: {max_rows_per_file} ')

        qs = self.queue_sizes()
        qs_results = qs[self.results_queue_name()]
        state = self.fetcher_state(qs)
        if qs_results == 0:
            kl.info(f'consolidate {self.name}: nothing to consolidate. State: {state}')
            return

        # get all messages from result queue
        remaining = qs_results
        while remaining > 0:
            messages_to_fetch = min(remaining, 120_000, max_queue_items_per_file)
            kl.debug(f'reading {messages_to_fetch} messages from results queue...')
            results: List[FetcherResult] = []
            while len(results) < messages_to_fetch:
                next_to_fetch = messages_to_fetch - len(results)
                new_results = self.pop_result_items(max_items=next_to_fetch)
                kl.debug(f'read {len(new_results)} new messages.')
                results.extend(new_results)
                if len(new_results) == 0:
                    break
            if len(results) == 0:
                break
            remaining -= len(results)
            # fetched_df = self.data_to_df(results)
            fetched_df = self.results_df(results)

            self.prepare_consolidation(fetched_df, max_rows_per_file=max_rows_per_file, **args)
            del fetched_df
            gc.collect()

            handles = [i.handle for i in results]
            ksqs.remove_messages(queue_name=self.results_queue_name(), receipt_handles=handles)

            del results
            gc.collect()

        kl.debug(f'consolidate {self.name}: finish.')

    def time_slice_ref(self, df: pd.DataFrame) -> Any:
        """identifier for time slice. by default, the latest timestamp"""
        return pd.to_datetime(df['capture_ts'].max())

    def time_slicing(self, df: pd.DataFrame) -> Generator[(pd.DataFrame, str, Any)]:
        """generator for time slicing. default is by capture date.
        Returns the dataframe slice, a string identifier, and an arbitrary object that will be used by save function"""
        dt_series = df['capture_ts'].str.slice(stop=10)
        dt_set = dt_series.unique()
        for dt_str in dt_set:
            dt_slice = df[dt_series == dt_str]
            slice_obj = self.time_slice_ref(dt_slice)
            yield dt_slice, dt_str, slice_obj

    def rows_slicing(self, df: pd.DataFrame, max_rows_per_file: int) -> Generator[(pd.DataFrame, int, int)]:
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
            decoded_data_str = decompress_str_base64(result_row['data'], result_row['compression'])
            decoded_data_list = orjson.loads(decoded_data_str) if decoded_data_str is not None else []
            data_rows = [self.prepare_row(result_row, decoded_item) for decoded_item in decoded_data_list]
            rows_accumulator.extend(data_rows)
            while len(rows_accumulator) >= max_rows_per_file:
                yield file_slice(), file_count, n_files

        while len(rows_accumulator) > 0:
            yield file_slice(), file_count, n_files

    @abstractmethod
    def prepare_consolidation(self, fetched_df: pd.DataFrame, max_rows_per_file: int, **args):

        if fetched_df is None or len(fetched_df) == 0:
            kl.info('empty dataframe, nothing to save.')

        kl.info(f'saving consolidated data for {len(fetched_df)} rows...')
        kprof = kp.KProfiler()
        kprof.log_mem('memory usage')

        # slice by table
        tables = set(fetched_df['table'].unique())
        for table in tables:
            table_slice = fetched_df.loc[fetched_df['table'] == table]
            kl.debug(f'preparing data for table {table} ({len(table_slice)} items)...')

            # slice by time
            for (time_slice_df, time_slice_id, time_slice_ref) in self.time_slicing(table_slice):

                # slice into files and prepare dataframe
                for (prepared_file_df, current_file, n_files) in self.rows_slicing(time_slice_df, max_rows_per_file):

                    self.save_consolidation(prepared_file_df, table, time_slice_id=time_slice_id,
                                            time_slice_ref=time_slice_ref,
                                            current_file=current_file, n_files=n_files, **args)
                    kprof.log_mem('memory usage before gc')
                    del prepared_file_df
                    gc.collect()
                    kprof.log_mem('memory usage after gc')

    @abstractmethod
    def save_consolidation(self, prepared_df: pd.DataFrame, table: str,
                           time_slice_id: str, time_slice_ref: Any,
                           current_file: int, n_files: int, **args):
        pass

    @abstractmethod
    def prepare_row(self, result_row: pd.Series, decoded_item: dict) -> dict:
        pass


class KarnakFetcherWorker:
    def __init__(self, fetcher: KarnakFetcher, extractor: str, n_threads: int = 1,
                 retries: int = 1,
                 loop_pause_seconds: int = 180):
        self.fetcher: KarnakFetcher = fetcher
        self.extractor = extractor
        self.n_threads = n_threads
        self.loop_pause_seconds = loop_pause_seconds
        self.retries = retries
        self.state = 'working'
        # self.state_check_lock = threading.RLock()

    def throttle_request(self):
        pass

    @abstractmethod
    def pop_work_queue_item(self, priority: Optional[int],
                            context: KarnakFetcherThreadContext, wait: bool) -> Optional[FetcherQueueItem]:
        pass

    @abstractmethod
    def pop_best_work_queue_item(self,
                                 context: KarnakFetcherThreadContext) -> Optional[FetcherQueueItem]:
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
    def fetch_item(self, item: FetcherQueueItem, context: KarnakFetcherThreadContext) -> FetcherResult:
        pass

    @abstractmethod
    def decide_failure_action(self, result: FetcherResult) -> (str, str):
        """Return:
                ('abort', None): do not try again.
                ('ignore', None): return message to queue.
                ('retry', None): retry in same strategy.
                ('retry', 'xpto'): move to extractor xpto.
                ('restart', 'xpto'): save in current extractor, and also reset errors and create new task in new extractor
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
            kl.debug(f"success fetching {result.queue_item.keys} in {result.elapsed_str()}, "
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

    def check_worker_state(self,) -> str:
        fetcher_state, working_priority = self.fetcher.fetcher_state()
        if fetcher_state == 'working':
            self.state = 'working'
        else:
            self.state = 'idle'
        return self.state

    def fetcher_thread_loop(self, thread_num: int):
        context = self.new_thread_context()
        while self.state == 'working':
            self.throttle_request()
            # self.state_check_lock.acquire()
            item = self.pop_best_work_queue_item(context=context)
            if item is None:
                kl.trace(f'thread {thread_num}: no item available in queue')
                self.state = 'idle'
            else:
                kl.trace(f'thread {thread_num}: read item from queue')
                self.process_item(item, context)
        kl.trace(f'thread {thread_num}: finished')

    def loop_pause(self):
        kl.trace(f'loop pause: {self.loop_pause_seconds} s')
        time.sleep(self.loop_pause_seconds)

    def worker_loop(self):
        while True:
            self.work()
            self.loop_pause()

    def work(self):
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
        _is_success = is_success if is_success is not None else (data is not None and error_type is None)
        _can_retry = can_retry if can_retry is not None else not _is_success
        _error_type = error_type if _is_success or error_type is not None else 'fetch-error'
        _error_message = error_message if _is_success or error_message is not None else 'fetch error'
        compressed_data = data

        try:
            if data is not None and compression is not None:
                compressed_data = self.fetcher.compress_data(data, compression=compression,
                                                             compression_level=compression_level)

        except Exception as e:
            kl.exception(f'compression error: exception for {item.table}, keys {item.keys}', e)
            _can_retry = False
            error_type = 'compression-error'
            error_message = str(e)

        fetcher_result = FetcherResult(item, data=compressed_data,
                                       capture_ts=capture_ts,
                                       rows=rows,
                                       elapsed=elapsed, is_success=_is_success,
                                       can_retry=_can_retry, compression=compression,
                                       error_type=error_type, error_message=error_message,
                                       response_edition=response_edition)
        return fetcher_result


class KarnakSqsFetcherWorker(KarnakFetcherWorker, ABC):

    def __init__(self, fetcher: KarnakSqsFetcher, extractor: str, n_threads: int = 1,
                 retries: int = 1,
                 loop_pause_seconds: int = 180):
        self.sqs_fetcher = fetcher
        super().__init__(fetcher=fetcher,
                         extractor=extractor,
                         n_threads=n_threads,
                         retries=retries,
                         loop_pause_seconds=loop_pause_seconds)

    def pop_best_work_queue_item(self,
                                 context: KarnakSqsFetcherThreadContext) -> Optional[FetcherQueueItem]:
        return self.sqs_fetcher.pop_best_work_queue_item(extractor=self.extractor,
                                                         context=context)

    def return_queue_item(self, item: FetcherQueueItem, context: KarnakSqsFetcherThreadContext):
        queue_name = self.sqs_fetcher.worker_queue_name(extractor=item.extractor, priority=item.priority)
        ksqs.return_message(queue_name, item.handle, sqs_client=context.sqs_client)

    def push_queue_item(self, item: FetcherQueueItem, context: KarnakSqsFetcherThreadContext):
        queue_name = self.sqs_fetcher.worker_queue_name(extractor=item.extractor, priority=item.priority)
        ksqs.send_messages(queue_name, [item.to_string()])

    def refresh_queue_item(self, item: FetcherQueueItem,
                           context: KarnakSqsFetcherThreadContext,
                           new_extractor: str = None,
                           new_priority: int = None):
        _extractor = item.extractor if new_extractor is None else new_extractor
        _priority = item.priority if new_priority is None else new_priority
        old_queue_name = self.sqs_fetcher.worker_queue_name(extractor=item.extractor, priority=item.priority)
        new_queue_name = self.sqs_fetcher.worker_queue_name(extractor=_extractor, priority=_priority)
        new_item = item
        if new_extractor is not None:
            new_item = item.set_extractor(extractor=new_extractor)
        if new_priority is not None:
            new_item = item.set_priority(priority=new_priority)
        ksqs.send_messages(new_queue_name, [new_item.to_string()])
        ksqs.remove_message(old_queue_name, item.handle)

    def complete_queue_item(self, result: FetcherResult, context: KarnakSqsFetcherThreadContext):
        """Put fetched results in queue (in case of success or failure with no retry)"""
        item = result.queue_item
        kl.trace(f'completing item: {item.keys}')
        message_str = result.to_string()
        worker_queue_name = self.sqs_fetcher.worker_queue_name(extractor=item.extractor, priority=item.priority)
        try:
            if len(message_str) > 262144:
                kl.warn(f'message too long to put in queue, key: {item.keys}, {len(item.to_string())} bytes')
            else:
                ksqs.send_messages(self.sqs_fetcher.results_queue_name(), [message_str])
            ksqs.remove_message(worker_queue_name, item.handle)
        except Exception as e:
            kl.exception(f'exception putting message in results queue: {item.keys}', e)
