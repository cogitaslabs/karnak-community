import threading
from typing import Optional, List, Dict, Any, Tuple
import boto3

from multiprocessing.pool import ThreadPool

import karnak3.core.log as kl
import karnak3.core.util as ku


_client_lock = threading.RLock()
_logger = kl.KLog(__name__)

queue_cache: Dict[str, str] = {}


# TODO REMOVE
def log_test():
    _logger.warn('warn-ok')


def get_client():
    with _client_lock:
        return boto3.client('sqs')


@ku.synchronized
def get_queue_url(queue_name: str, sqs_client=None, use_cache: bool = True) -> (str, Any):
    _sqs_client = sqs_client if sqs_client is not None else get_client()
    queue_url = queue_cache.get(queue_name) if use_cache else None
    if queue_url is None:
        response = _sqs_client.get_queue_url(QueueName=queue_name)
        queue_url = response['QueueUrl']
        queue_cache[queue_name] = queue_url
    return queue_url, _sqs_client


def remove_message(queue_name: str, receipt_handle: str, sqs_client=None):
    queue_url, sqs_client = get_queue_url(queue_name, sqs_client)
    sqs_client.delete_message(QueueUrl=queue_url, ReceiptHandle=receipt_handle)


def remove_messages(queue_name: str, receipt_handles: List[str],
                    sqs_client=None, threads: int = 1):
    queue_url, sqs_client = get_queue_url(queue_name, sqs_client)
    chunks = [receipt_handles[i:i+10] for i in range(0, len(receipt_handles), 10)]

    def chunk_to_entries(_chunk: List[str]) -> List[Dict[str, str]]:
        entries_l = [{'Id': str(i), 'ReceiptHandle': _chunk[i]} for i in range(0, len(_chunk))]
        return entries_l

    if threads == 1 or len(receipt_handles) <= 100:
        queue_url, sqs_client = get_queue_url(queue_name, sqs_client)
        for chunk in chunks:
            entries = chunk_to_entries(chunk)
            sqs_client.delete_message_batch(QueueUrl=queue_url, Entries=entries)
    else:
        def delete_batch(_chunk: List[str]):
            # client is not thread safe.
            # TODO to avoid creating a client for every chunk,
            #  rewrite using a Queue and not a Pool.
            sqs_client_t = get_client()
            _entries = chunk_to_entries(_chunk)
            sqs_client_t.delete_message_batch(QueueUrl=queue_url, Entries=_entries)
            # _logger.trace(f'thread removed {len(_entries)} messages from queue.')
        pool = ThreadPool(threads)
        pool.map(delete_batch, chunks)
        _logger.debug(f'removed {len(receipt_handles)} messages from queue.')


def return_message(queue_name: str, receipt_handle: str, sqs_client=None):
    queue_url, sqs_client = get_queue_url(queue_name, sqs_client)
    sqs_client.change_message_visibility(QueueUrl=queue_url, ReceiptHandle=receipt_handle,
                                         VisibilityTimeout=0)


def return_messages(queue_name: str, receipt_handles: List[str], sqs_client=None):
    queue_url, sqs_client = get_queue_url(queue_name, sqs_client)
    # TODO implement batch version
    for receipt_handle in receipt_handles:
        sqs_client.return_message(QueueUrl=queue_url, ReceiptHandle=receipt_handle)


# def send_messages(queue_name: str,
#                   messages: list,
#                   group_id: Optional[str] = None,
#                   sqs_client=None,
#                   threads: int = 1) -> list:
#     assert threads > 0
#     if threads == 1:
#         return _send_messages_single_thread(queue_name=queue_name,
#                                             messages=messages,
#                                             group_id=group_id,
#                                             sqs_client=sqs_client)
#     else:
#         return _send_messages_multi_thread(queue_name=queue_name,
#                                            messages=messages,
#                                            group_id=group_id)


def send_messages(queue_name: str,
                  messages: list,
                  group_id: Optional[str] = None,
                  sqs_client=None,
                  threads: int = 1) -> list:
    """
    Returns: list os failed ids
    """
    failed_indexes: List[int] = []
    next_message = 0

    @ku.synchronized
    def pop_next_slice_index(n: int = 10) -> int:
        """returns index of first message in next slice, for a slice of n messages"""
        nonlocal next_message
        next_slice_index = next_message
        next_message += n
        return next_slice_index

    # @ku.synchronized
    # def pop_messages(n: int = 10) -> Tuple[list, int]:
    #     """returns (list of messages, first_message_index)"""
    #     nonlocal next_message
    #     messages_slice = messages[next_message: next_message+n]
    #     first_message_index = next_message
    #     next_message += n
    #     return messages_slice, first_message_index

    @ku.synchronized
    def push_failed_indexes(idxs: List[int]):
        nonlocal failed_indexes
        failed_indexes.extend(idxs)

    def send_slice(_sqs_client, queue_url) -> int:
        """return umber of messages sent (with success or not)"""
        n = 10  # max of messages to get
        next_slice_index = pop_next_slice_index(n)
        messages_slice = messages[next_slice_index: next_slice_index + n]
        if messages_slice is None or len(messages_slice) == 0:
            return 0
        batch_items = []
        for j in range(len(messages_slice)):
            item = {'Id': str(next_slice_index + j), 'MessageBody': messages_slice[j]}
            if group_id is not None:
                item['MessageGroupId'] = group_id
            batch_items.append(item)
        result = _sqs_client.send_message_batch(QueueUrl=queue_url, Entries=batch_items)
        if result.get('Failed'):
            slice_failed_indexes = [int(item['Id']) for item in result.get('Failed')]
            if len(slice_failed_indexes) > 0:
                push_failed_indexes(slice_failed_indexes)
        return len(batch_items)

    def sender_worker(_sqs_client=None):
        queue_url, _sqs_client = get_queue_url(queue_name, _sqs_client)
        keep_going = True
        while keep_going:
            keep_going = send_slice(_sqs_client, queue_url) > 0

    if threads <= 1:
        sender_worker(sqs_client)
    else:
        kl.debug(f'putting {len(messages)} messages in queue with {threads} threads')
        thread_list = []
        for i in range(threads):
            t = threading.Thread(target=sender_worker)
            t.start()
            thread_list.append(t)
        for t in thread_list:
            t.join()

    failed_messages = [messages[i] for i in failed_indexes]
    return failed_messages


def receive_messages(queue_name: str,
                     max_messages: int = 10,
                     wait_seconds: int = 0,
                     sqs_client=None,
                     threads: int = 1) -> Dict[(str, str)]:
    if threads == 1 or max_messages <= 500:
        result_messages = _receive_messages(queue_name=queue_name,
                                            max_messages=max_messages,
                                            wait_seconds=wait_seconds,
                                            sqs_client=sqs_client)
    else:
        result_messages = {}

        @ku.synchronized
        def push_messages(messages: Dict[(str, str)]):
            nonlocal result_messages
            result_messages.update(messages)

        def reader_worker(max_msg: int):
            d = _receive_messages(queue_name=queue_name,
                                  max_messages=max_msg,
                                  wait_seconds=wait_seconds,
                                  sqs_client=None)
            push_messages(d)

        _logger.debug(f'reading {max_messages} messages from queue with {threads} threads')
        thread_list = []
        max_messages_1 = max_messages // threads
        max_messages_0 = max_messages - (max_messages_1 * (threads - 1))
        for i in range(threads):
            max_messages_t = max_messages_1 if i > 0 else max_messages_0
            t = threading.Thread(target=reader_worker, args=(max_messages_t,))
            t.start()
            thread_list.append(t)
        for t in thread_list:
            t.join()

    _logger.trace(f'sqs: {len(result_messages)} messages read from {queue_name}')
    return result_messages


def _receive_messages(queue_name: str,
                     max_messages: int = 10,
                     wait_seconds: int = 0,
                     sqs_client=None) -> Dict[(str, str)]:
    """Returns a dict of {message_handle: body}"""
    queue_url, sqs_client = get_queue_url(queue_name, sqs_client)

    remaining_messages = max_messages
    result_messages = {}

    while remaining_messages > 0:
        request_messages = min(10, remaining_messages)
        result = sqs_client.receive_message(QueueUrl=queue_url,
                                            MaxNumberOfMessages=request_messages,
                                            WaitTimeSeconds=wait_seconds)
        remaining_messages -= request_messages
        if result.get('Messages'):
            new_results = {m['ReceiptHandle']: m['Body'] for m in result['Messages']}
            result_messages.update(new_results)
            # _logger.trace(f'sqs: {len(new_results)} messages read from {queue_name}')
        else:
            # _logger.trace(f'sqs: 0 messages read from {queue_name}')
            break

    # _logger.trace(f'sqs: {len(result_messages)} messages read from {queue_name}')
    return result_messages


def queue_attributes(queue_name: str, sqs_client=None) -> dict:
    queue_url, sqs_client = get_queue_url(queue_name, sqs_client)
    result = sqs_client.get_queue_attributes(QueueUrl=queue_url, AttributeNames=['All'])
    return result['Attributes']


def purge_queue(queue_name: str, sqs_client=None):
    queue_url, sqs_client = get_queue_url(queue_name, sqs_client)
    result = sqs_client.purge_queue(QueueUrl=queue_url)
