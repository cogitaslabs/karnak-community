from __future__ import annotations

from abc import ABC
from typing import Generator, Any
import gc

import karnak3.cloud.aws.sqs as ksqs
import karnak3.core.profiling as kp
from karnak3.core.util import synchronized
from karnak3.fetch.base_fetcher import *


class KarnakSqsFetcherThreadContext(KarnakFetcherThreadContext):
    def __init__(self):
        self.sqs_client = ksqs.get_client()
        super().__init__()


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

    def populate_worker_queue(self, items: List[FetcherQueueItem],
                              extractor: str,
                              priority: Optional[int]):
        worker_queue_name = self.worker_queue_name(extractor=extractor, priority=priority)
        kl.trace(f'putting {len(items)} messages in queue {worker_queue_name}')
        contents = [i.to_string() for i in items]
        ksqs.send_messages(worker_queue_name, contents)

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
        items = ksqs.receive_messages(queue_name=queue_name, max_messages=1,
                                      wait_seconds=wait_seconds,
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
                                 context: KarnakSqsFetcherThreadContext) \
            -> Optional[FetcherQueueItem]:
        priorities = self.priorities()
        for retry in [0, 1]:  # two rounds of attempts
            for p in priorities:
                queue_name = self.worker_queue_name(extractor, priority=p)
                # only checks empty in first round.
                if retry or not self.is_empty_queue(queue_name):
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
        ret = [FetcherResult.from_string(items[handle], handle=handle)
               for handle in items if items is not None]
        return ret

    def consolidate(self, max_queue_items_per_file: int = 120_000,
                    max_rows_per_file: int = 2_000_000, **args):
        kl.info(f'consolidate {self.name}: start.')
        kl.debug(f'max_queue_items_per_file: {max_queue_items_per_file}, '
                 f'max_rows_per_file: {max_rows_per_file} ')

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
            rows_accumulator.extend(data_rows)
            while len(rows_accumulator) >= max_rows_per_file:
                yield file_slice(), file_count, n_files

        while len(rows_accumulator) > 0:
            yield file_slice(), file_count, n_files

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
                for (prepared_file_df, current_file, n_files) \
                        in self.rows_slicing(time_slice_df, max_rows_per_file):

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

    def new_thread_context(self) -> KarnakSqsFetcherThreadContext:
        ctx = KarnakSqsFetcherThreadContext()
        return ctx

    def pop_best_work_queue_item(self,
                                 context: KarnakSqsFetcherThreadContext) \
            -> Optional[FetcherQueueItem]:
        return self.sqs_fetcher.pop_best_work_queue_item(extractor=self.extractor,
                                                         context=context)

    def return_queue_item(self, item: FetcherQueueItem, context: KarnakSqsFetcherThreadContext):
        queue_name = self.sqs_fetcher.worker_queue_name(extractor=item.extractor,
                                                        priority=item.priority)
        ksqs.return_message(queue_name, item.handle, sqs_client=context.sqs_client)

    def push_queue_item(self, item: FetcherQueueItem, context: KarnakSqsFetcherThreadContext):
        queue_name = self.sqs_fetcher.worker_queue_name(extractor=item.extractor,
                                                        priority=item.priority)
        ksqs.send_messages(queue_name, [item.to_string()])

    def refresh_queue_item(self, item: FetcherQueueItem,
                           context: KarnakSqsFetcherThreadContext,
                           new_extractor: str = None,
                           new_priority: int = None):
        _extractor = item.extractor if new_extractor is None else new_extractor
        _priority = item.priority if new_priority is None else new_priority
        old_queue_name = self.sqs_fetcher.worker_queue_name(extractor=item.extractor,
                                                            priority=item.priority)
        new_queue_name = self.sqs_fetcher.worker_queue_name(extractor=_extractor,
                                                            priority=_priority)
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
        worker_queue_name = self.sqs_fetcher.worker_queue_name(extractor=item.extractor,
                                                               priority=item.priority)
        try:
            if len(message_str) > 262144:
                kl.warn(f'message too long to put in queue, key: {item.keys}, '
                        f'{len(item.to_string())} bytes')
            else:
                ksqs.send_messages(self.sqs_fetcher.results_queue_name(), [message_str])
            ksqs.remove_message(worker_queue_name, item.handle)
        except Exception as e:
            kl.exception(f'exception putting message in results queue: {item.keys}', e)
