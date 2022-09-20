from __future__ import annotations

from abc import ABC
from typing import Iterable

import karnak3.cloud.aws.sqs as ksqs
from karnak3.core.util import synchronized
from karnak3.fetch.base_fetcher import *


class KarnakSqsFetcherThreadContext(KarnakFetcherThreadContext):
    def __init__(self, context_vars: dict = None):
        self.sqs_client = ksqs.get_client()
        super().__init__(context_vars=context_vars)


class KarnakSqsFetcher(KarnakFetcher):
    def __init__(self, name: str,
                 tables: List[str],
                 environment: str,
                 extractors: Optional[List[str]] = None,
                 max_priority: Optional[int] = None,
                 empty_work_queue_recheck_seconds: int = 300):
        super().__init__(name, tables, environment, extractors, max_priority)
        self.empty_queue_control: Dict[str, datetime.datetime] = {}
        self.empty_queue_counter: Dict[str, int] = {}
        self.default_sqs_client = ksqs.get_client()
        self.empty_work_queue_recheck_seconds = empty_work_queue_recheck_seconds

        # helper series counters for consolidation
        self.table_consolidation_full_handle_cnt: Optional[pd.Series] = None
        self.table_consolidation_processed_handle_cnt: Optional[pd.Series] = None

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

    def _queue_size(self, queue_name: str, sqs_client=None) -> int:
        attr = ksqs.queue_attributes(queue_name, sqs_client=sqs_client)
        available = int(attr['ApproximateNumberOfMessages'])
        in_flight = int(attr['ApproximateNumberOfMessagesNotVisible'])
        delayed = int(attr['ApproximateNumberOfMessagesDelayed'])
        return available + in_flight + delayed

    def queue_sizes(self, sqs_client=None) -> Dict[str, int]:
        """Returns approximate message count for all queues."""
        kl.trace(f'getting queue sizes')
        _sqs_client = sqs_client if sqs_client is not None else self.default_sqs_client
        qs = {}
        queue_names = self.worker_queue_names() + [self.results_queue_name()]
        for qn in queue_names:
            qs[qn] = self._queue_size(qn)
        return qs

    #
    # purge
    #

    def purge(self,
              priority: Optional[int] = None,
              extractor: Optional[str] = None,
              purge_all: bool = None):

        if purge_all:
            self.purge_all_queues()
        else:
            if extractor is None:
                kl.error('purge not performed: extractor not defined.')
            else:
                self.purge_worker_queue(extractor, priority)
    #
    # kickoff
    #

    def populate_worker_queue(self, items: List[FetcherQueueItem],
                              extractor: str,
                              priority: Optional[int],
                              threads: int = 1):
        worker_queue_name = self.worker_queue_name(extractor=extractor, priority=priority)
        kl.trace(f'putting {len(items)} messages in queue {worker_queue_name}')
        contents = [i.to_string() for i in items]
        ksqs.send_messages(worker_queue_name, contents, threads=threads)

    #
    # worker
    #

    def create_thread_context(self) -> KarnakSqsFetcherThreadContext:
        ctx = KarnakSqsFetcherThreadContext()
        return ctx

    @synchronized
    def set_empty_queue(self, queue_name: str):
        self.empty_queue_control[queue_name] = datetime.datetime.now(tz=pytz.utc)
        eq_counter = self.empty_queue_counter.get(queue_name, 0)
        self.empty_queue_counter[queue_name] = eq_counter + 1
        kl.trace(f'queue {queue_name} marked empty (counter={eq_counter + 1})')

    @synchronized
    def set_not_empty_queue(self, queue_name: str):
        self.empty_queue_counter[queue_name] = 0

    @synchronized
    def is_empty_queue(self, queue_name: str) -> bool:
        _counter_threshold = 4  # only after _counter_threshold attempts this funtion will return true
        eq_counter = self.empty_queue_counter.get(queue_name, 0)
        if eq_counter < _counter_threshold:
            return False
        eq_time = self.empty_queue_control.get(queue_name)
        assert eq_time is not None
        now = datetime.datetime.now(tz=pytz.utc)
        if now - eq_time >= datetime.timedelta(seconds=self.empty_work_queue_recheck_seconds):
            self.set_not_empty_queue(queue_name)
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
            # kl.trace(f'queue {queue_name} may be set as empty')
            self.set_empty_queue(queue_name)
            return None
        else:
            assert len(items) == 1
            self.set_not_empty_queue(queue_name)
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

    def pop_result_items(self, max_items: int,
                         threads: int = 1) -> List[FetcherResult]:
        items = ksqs.receive_messages(queue_name=self.results_queue_name(),
                                      max_messages=max_items, wait_seconds=20,
                                      threads=threads)
        ret = [FetcherResult.from_string(items[handle], handle=handle)
               for handle in items if items is not None]
        return ret

    def consolidate(self, max_queue_items_per_file: int = 120_000,
                    max_rows_per_file: int = 2_000_000,
                    threads: int = 1,
                    **args):
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
            fetch_cnt = 0
            MAX_MESSAGE_FETCH_RETRIES = 5
            while len(results) < messages_to_fetch and fetch_cnt < MAX_MESSAGE_FETCH_RETRIES:
                next_to_fetch = messages_to_fetch - len(results)
                new_results = self.pop_result_items(max_items=next_to_fetch,
                                                    threads=threads)
                kl.debug(f'read {len(new_results)} new messages.')
                results.extend(new_results)
                if len(new_results) == 0:
                    break
                fetch_cnt += 1
            if len(results) == 0:
                break
            remaining -= len(results)
            # fetched_df = self.data_to_df(results)
            fetched_df = self.results_df(results)

            self.prepare_consolidation(fetched_df, max_rows_per_file=max_rows_per_file,
                                       threads=threads, **args)
            del fetched_df
            # gc.collect()

            # handles = [i.handle for i in results]
            # kl.debug(f'removing {len(handles)} messages from results queue...')
            # ksqs.remove_messages(queue_name=self.results_queue_name(), receipt_handles=handles)

            del results
            gc.collect()

        kl.debug(f'consolidate {self.name}: finish.')

    def prepare_table_consolidation(self, table_slice_df: pd.DataFrame):
        self.table_consolidation_full_handle_cnt = table_slice_df[['handle', 'rows']]\
            .groupby('handle').sum('rows')['rows']
        self.table_consolidation_processed_handle_cnt = self.table_consolidation_full_handle_cnt.copy()
        self.table_consolidation_processed_handle_cnt[:] = 0

    def clean_slice_consolidation(self, prepared_file_df: pd.DataFrame,
                                  table: str, threads: int = 1):
        # count handles processed and find those fully processed to remove
        slice_handle_cnt = prepared_file_df['handle'].value_counts()
        table_handle_cnt_new = \
            self.table_consolidation_processed_handle_cnt.add(slice_handle_cnt, fill_value=0)
        table_handle_cnt_missing = \
            self.table_consolidation_full_handle_cnt.sub(table_handle_cnt_new)
        handles_fulfilled = table_handle_cnt_missing[table_handle_cnt_missing == 0]
        handles_to_remove = handles_fulfilled.index.intersection(slice_handle_cnt.index)
        handles = list(handles_to_remove)

        # handles = list(prepared_file_df['handle'].unique())
        kl.debug(f'removing {len(handles)} messages from results queue...')
        ksqs.remove_messages(queue_name=self.results_queue_name(), receipt_handles=handles,
                             threads=threads)
        self.table_consolidation_processed_handle_cnt = table_handle_cnt_new

    #
    # purge
    #

    def _purge_worker_queues(self, queue_names: Iterable[str]):
        for qn in queue_names:
            qs = self._queue_size(qn)
            kl.info(f'purging queue {qn} which had {qs} messages...')
            ksqs.purge_queue(qn)

    def purge_worker_queue(self, extractor: str, priority: Optional[int]):
        queue_name = self.worker_queue_name(extractor=extractor, priority=priority)
        self._purge_worker_queues([queue_name])

    def purge_all_queues(self):
        queue_names = self.worker_queue_names(extractor=None)
        self._purge_worker_queues(queue_names)


class KarnakSqsFetcherWorker(KarnakFetcherWorker, ABC):

    def __init__(self, fetcher: KarnakSqsFetcher, extractor: str, n_threads: int = 1,
                 retries: int = 1,
                 loop_pause_seconds: int = 180,
                 stop_after_queue_items: Optional[int] = None,
                 stop_after_minutes: Optional[int] = None
                 ):
        self.sqs_fetcher = fetcher
        super().__init__(fetcher=fetcher,
                         extractor=extractor,
                         n_threads=n_threads,
                         retries=retries,
                         loop_pause_seconds=loop_pause_seconds,
                         stop_after_queue_items=stop_after_queue_items,
                         stop_after_minutes=stop_after_minutes)

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
        kl.trace(f'completing item for table {item.table}: {item.keys}')
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
