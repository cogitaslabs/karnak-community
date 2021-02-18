import karnak.util.log as kl
import karnak.util.aws.sqs as ksqs

from abc import abstractmethod
from typing import Dict, List, Optional
import datetime
import time
import pandas as pd
import json
import threading


def json_serial(obj):
    """JSON serializer for objects not serializable by default json code"""
    if isinstance(obj, (datetime.datetime, datetime.date)):
        return obj.isoformat()
    raise TypeError("Type %s not serializable" % type(obj))


class FetcherItem:
    def __init__(self, key: str, start_ts: datetime.datetime, strategy: Optional[str] = None,
                 current_retries: int = 0,
                 is_success: bool = False, is_timeout: bool = False, content=None, handle=None,
                 fetch_errors=None):
        self.key: str = key
        self.start_ts: datetime.datetime = start_ts
        self.strategy: Optional[str] = strategy
        self.current_retries: int = current_retries
        self.is_success: bool = is_success
        self.is_timeout: bool = is_timeout
        self.content = content
        self.fetch_errors = fetch_errors if fetch_errors is not None else []
        self.handle = handle

    def to_string(self):
        filtered_dict = self.__dict__.copy()
        del filtered_dict['handle']

        js = json.dumps(filtered_dict, default=json_serial)
        return js

    @classmethod
    def from_string(cls, string, handle=None):
        d = json.loads(string)
        try:
            f = FetcherItem(key=d['key'],
                            start_ts=datetime.datetime.fromisoformat(d['start_ts']),
                            strategy=d['strategy'],
                            current_retries=d['current_retries'],
                            is_success=d['is_success'],
                            is_timeout=d['is_timeout'],
                            content=d['content'],
                            fetch_errors=d['fetch_errors'],
                            handle=handle)
        except KeyError:
            msg = string.replace("\n", "")
            kl.error(f'invalid message: {msg}')
            return None
        return f


class SqsFetcher:
    def __init__(self, name: str,
                 results_queue: str,
                 worker_queues: List[str],
                 strategies: List[str] = None,
                 staging: bool = False):
        assert len(worker_queues) > 0
        self.name = name
        self.strategies = strategies if strategies is not None else ['default']
        self.is_multi_strategy = len(strategies) > 1
        self.staging = staging

        queue_prefix = ''
        if staging:
            queue_prefix = 'staging-'
        self.worker_queues = [queue_prefix + q for q in worker_queues]
        self.results_queue = queue_prefix + results_queue

    #
    # general
    #

    def queue_sizes(self, wait_seconds: Optional[int] = None, sqs_client=None) -> Dict[str, int]:
        """Returns approximate message count for all queues. Retries if any is zeroed."""
        # FIXME wait_seconds should be 60
        if wait_seconds is None:
            wait_seconds = 2

        i = 0
        qs_results = 0
        qs_workers = 0
        retries = 2
        kl.trace(f'getting queue sizes (wait up to {wait_seconds * (retries+1)} s)')
        qs = {}
        queues = [self.results_queue] + self.worker_queues
        while i < retries and (qs_results == 0 or qs_workers == 0):
            for q in queues:
                attr = ksqs.queue_attributes(q, sqs_client=sqs_client)
                available = int(attr['ApproximateNumberOfMessages'])
                in_flight = int(attr['ApproximateNumberOfMessagesNotVisible'])
                delayed = int(attr['ApproximateNumberOfMessagesDelayed'])
                qs[q] = available + in_flight + delayed
            qs_results = qs[self.results_queue]
            qs_workers = sum([qs[queue_name]for queue_name in self.worker_queues])
            i += 1
            # sleep and retry if any queue has 0 elements
            if i < retries and (qs_results == 0 or qs_workers == 0):
                time.sleep(wait_seconds)
        qs_str = ', '.join([f'{q}: {qs[q]}' for q in qs])
        # kl.trace('queue sizes: ' + qs_str)
        return qs

    def fetcher_state(self, qs: Optional[Dict[str, int]] = None, sqs_client=None, wait_seconds: Optional[int] = None) -> str:
        """Returns current fetcher state: processing, consolidating, idle."""

        if qs is None:
            qs = self.queue_sizes(sqs_client=sqs_client, wait_seconds=wait_seconds)
        qs_results = qs[self.results_queue]
        qs_workers = sum([qs[queue_name]for queue_name in self.worker_queues])

        if qs_results + qs_workers == 0:
            return 'idle'
        elif qs_workers == 0:
            return 'consolidation'
        else:
            return 'working'

    def worker_queue(self, strategy: str = None) -> Optional[str]:
        if strategy is None:
            return self.worker_queues[0]
        elif strategy in self.strategies:
            return self.worker_queues[self.strategies.index(strategy)]
        else:
            kl.error(f'invalid worker strategy: {strategy}')
            return None

    #
    # queue access
    #

    @classmethod
    def fetch_items(cls, queue_name: str, max_items=1, sqs_client=None) -> List[FetcherItem]:
        """Returns: dict of message handle, FetcherItem"""
        items = ksqs.receive_messages(queue_name=queue_name, max_messages=max_items, sqs_client=sqs_client)
        ret = [FetcherItem.from_string(items[handle], handle=handle) for handle in items if items is not None]
        return ret

    #
    # kickoff
    #

    @ abstractmethod
    def keys_to_fetch(self, max_fetch: Optional[int] = None, force_fetch: Optional[List[str]] = None) -> List[str]:
        return []

    def set_initial_strategy(self, df: pd.DataFrame, strategies: Optional[List[str]] = None):
        valid_strategies = strategies
        if strategies is None:
            valid_strategies = self.strategies
        df['strategy'] = valid_strategies[0]

    @classmethod
    def build_items_list(cls, keys_strategies: Dict[(str, Optional[str])], ref_ts: datetime.datetime)\
            -> List[FetcherItem]:
        return [FetcherItem(key=k, start_ts=ref_ts) for k in keys_strategies]

    def populate_worker_queue(self, items: List[FetcherItem], strategy: str):
        contents = [i.to_string() for i in items]
        ksqs.send_messages(self.worker_queue(strategy=strategy), contents)

    def kickoff(self, max_fetch: Optional[int] = None, force_fetch: Optional[List[str]] = None,
                strategies: Optional[List[str]] = None, force=False) -> bool:
        # test fetch state
        state = self.fetcher_state()
        if state != 'idle' and not force:
            kl.info(f'cannot kickoff {self.name}: current state is {state}.')
            return False

        # keys and initial strategies
        df = pd.DataFrame(self.keys_to_fetch(max_fetch=max_fetch, force_fetch=force_fetch), columns={'key'})
        if len(df) == 0:
            kl.info(f'cannot kickoff {self.name}: 0 ids to fetch.')
            return False
        self.set_initial_strategy(df, strategies=strategies)

        ref_ts = datetime.datetime.now()
        ref_ts_str = ref_ts.strftime('%Y%m%d_%H%M%S')
        kl.debug(f'populating {self.name} with {len(df)} elements, ref {ref_ts_str}.')

        def row_to_item(row):
            return FetcherItem(key=row['key'], start_ts=ref_ts, strategy=row['strategy'])
        df['item'] = df.apply(row_to_item, axis=1)

        for i in range(len(self.strategies)):
            strategy = self.strategies[i]
            df_strategy = df[df['strategy'] == strategy]
            self.populate_worker_queue(df_strategy['item'].tolist(), strategy)

        kl.debug(f'kickoff completed for {self.name}, ref {ref_ts_str}.')

    #
    # consolidator
    #

    @ abstractmethod
    def save_results(self, df: pd.DataFrame, strategy: str, ref_ts: datetime.datetime,
                     current_file: int, n_files: int,
                     output_folder: str, local_only: bool):
        pass

    @abstractmethod
    def save_consolidated(self, fetched_df: pd.DataFrame, **args):
        pass

    def consolidate(self, **args):
        kl.debug(f'consolidate {self.name}: start.')

        qs = self.queue_sizes()
        qs_results = qs[self.results_queue]
        state = self.fetcher_state(qs)
        if qs_results == 0:
            kl.info(f'consolidate {self.name}: nothing to consolidate. State: {state}')
            return

        # get all messages from result queue
        # TODO: improvement: interactive algorithm that gets less elements each time
        remaining = qs_results
        while remaining > 0:
            messages_to_fetch = max(remaining, 100_000)
            items = self.fetch_items(self.results_queue, max_items=messages_to_fetch)
            if len(items) == 0:
                break
            remaining -= len(items)
            fetched_data = [i.content for i in items]
            fetched_df = pd.DataFrame(fetched_data)

            self.save_consolidated(fetched_df, **args)

            handles = [i.handle for i in items]
            ksqs.remove_messages(queue_name=self.results_queue, receipt_handles=handles)

        kl.debug(f'consolidate {self.name}: finish.')

    #
    # kill
    #

    def purge(self, workers: List[str], results: bool = False):
        kl.debug(f'purge {self.name}: start.')

        if workers is None and not results:
            kl.info(f'purge: no action defined - nothing to do!')
            return

        qs = self.queue_sizes()
        state = self.fetcher_state(qs)
        if state == 'idle':
            kl.info(f'purge: fetcher state {state}, nothing to do')
            return

        items = []
        if workers is not None:
            items.extend(workers)
        if results:
            items.append('results')

        for w in items:
            if w == 'results':
                queue = self.results_queue
            else:
                queue = self.worker_queue(w)
            n = qs[queue]
            if n > 0:
                kl.info(f"queue for '{w}' ({queue}) has {n} messages: purging")
                ksqs.purge_queue(queue)
            else:
                kl.info(f"queue for '{w}' ({queue}) has {n} messages: nothing to do")

        kl.debug(f'purge {self.name}: finish.')


class FetcherResult:
    def __init__(self, key: str, results: Optional[dict], elapsed: datetime.datetime,
                 can_retry: bool = False, error_type: str = None, error_message: str = None):
        self.key = key
        self.results = results
        self.elapsed = elapsed
        self.can_retry = can_retry
        self.error_type = error_type
        self.error_message = error_message

        self.is_success = self.results is not None

    def elapsed_str(self):
        return str(self.elapsed)


class SqsFetcherWorker:
    def __init__(self, fetcher: SqsFetcher, strategy: str, n_threads: int = 1, items_per_request: int = 1,
                 loop_pause_seconds: int = 180):
        self.fetcher = fetcher
        self.strategy = strategy
        self.worker_queue_name = self.fetcher.worker_queue(self.strategy)
        self.items_per_request = items_per_request
        self.n_threads = n_threads
        self.loop_pause_seconds = loop_pause_seconds

        self.state = 'idle'
        self.state_check_lock = threading.RLock()

    @abstractmethod
    def throttle_request(self):
        pass

    def fetch_items(self, max_items: int = 1, sqs_client=None) -> List[FetcherItem]:
        self.state_check_lock.acquire()
        ret = self.fetcher.fetch_items(self.worker_queue_name, max_items=max_items, sqs_client=sqs_client)
        self.state_check_lock.release()
        return ret

    def return_item(self, item: FetcherItem):
        strategy = item.strategy
        queue = self.fetcher.worker_queue(strategy)
        ksqs.return_message(self.worker_queue_name, item.handle)

    def send_item(self, item: FetcherItem):
        strategy = item.strategy
        queue = self.fetcher.worker_queue(strategy)
        ksqs.send_messages(queue, [item.to_string()])
    
    def resend_item(self, item: FetcherItem):
        strategy = item.strategy
        queue = self.fetcher.worker_queue(strategy)
        ksqs.send_messages(queue, [item.to_string()])
        ksqs.remove_message(self.worker_queue_name, item.handle)

    def complete_item(self, item: FetcherItem):
        """Put fetcher item in results queue (in case of success or non-retryable failure)"""
        ksqs.send_messages(self.fetcher.results_queue, [item.to_string()])
        ksqs.remove_message(self.worker_queue_name, item.handle)

    @abstractmethod
    def fetch_key(self, key: str) -> FetcherResult:
        pass

    def fetch_batch(self, key_batch: List[str]) -> List[FetcherResult]:
        # default implementation
        ret = []
        for key in key_batch:
            ret.append(self.fetch_key(key))
        return ret

    @abstractmethod
    def item_to_data(self, result: FetcherResult, item: FetcherItem) -> dict:
        pass

    @abstractmethod
    def decide_failure_action(self, item: FetcherItem, result: FetcherResult) -> (str, str):
        """Return:
                ('abort', None): do not try again.
                ('ignore', None): return message to queue.
                ('retry', None): retry in same strategy.
                ('retry', 'xpto'): move to strategy xpto.
                ('restart', 'xpto'): save in current strategy, and also reset errors and create new task in new strategy
        """
        return 'abort', None

    def process_batch(self, items: List[FetcherItem]):
        # TODO what happens if we have a duplicate key?
        key_batch = [i.key for i in items]
        results = self.fetch_batch(key_batch)
        for result in results:
            item = items[key_batch.index(result.key)]  # find item
            # fill FetcherItem info and add metadata
            data = self.item_to_data(result, item)
            item.content = data
            if result.is_success:
                # successful ones: move to results
                kl.debug(f"success fetching {result.key} in {result.elapsed_str()}, attempt {item.current_retries}")
                item.content = data
                self.complete_item(item)
            else:  # failure
                action, new_strategy = self.decide_failure_action(item, result)
                if action == 'abort':
                    self.complete_item(item)
                elif action == 'ignore':
                    self.return_item(item)
                elif action == 'restart':
                    new_item = FetcherItem(item.key, item.start_ts, new_strategy)
                    self.send_item(new_item)
                    self.complete_item(item)
                else:  # retry
                    item.current_retries += 1
                    if new_strategy is not None and new_strategy != self.strategy:
                        item.strategy = new_strategy
                    self.resend_item(item)

    def worker_queue(self) -> str:
        return self.fetcher.worker_queue(self.strategy)

    def check_worker_state(self, force_recheck=False, sqs_client=None, wait_seconds: Optional[int] = None) -> str:
        self.state_check_lock.acquire()
        if force_recheck or self.state == 'working':
            qs = self.fetcher.queue_sizes(sqs_client=sqs_client, wait_seconds=wait_seconds)
            qs_workers = qs[self.worker_queue()]

            if qs_workers == 0:
                self.state = 'done'
            else:
                self.state = 'working'
        self.state_check_lock.release()
        return self.state

    def fetcher_thread_loop(self, thread_num: int):
        sqs_client = ksqs.get_client()
        while self.state == 'working':
            self.throttle_request()
            items = self.fetch_items(self.items_per_request, sqs_client=sqs_client)
            kl.trace(f'thread {thread_num}: read {len(items)} items from queue')
            if len(items) == 0:
                self.check_worker_state(force_recheck=False, sqs_client=sqs_client, wait_seconds=20)
            else:
                self.process_batch(items)
        kl.trace(f'thread {thread_num}: finished')

    def loop_pause(self):
        kl.trace(f'loop pause: {self.loop_pause_seconds} s')
        time.sleep(self.loop_pause_seconds)

    def worker_loop(self):
        while True:
            self.work()
            self.loop_pause()

    def work(self):
        self.check_worker_state(force_recheck=True)
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

        kl.info(f'worker for {self.strategy} finished: fetcher state {self.state}')



