import boto3
import threading
from typing import Optional, List, Dict, Any


client_lock = threading.RLock()


def get_client():
    client_lock.acquire()
    ret = boto3.client('sqs')
    client_lock.release()
    return ret


def get_queue_url(queue_name: str, sqs_client=None) -> (str, Any):
    if sqs_client is None:
        sqs_client = get_client()
    response = sqs_client.get_queue_url(QueueName=queue_name)
    return response['QueueUrl'], sqs_client


def remove_message(queue_name: str, receipt_handle: str, sqs_client=None):
    queue_url, sqs_client = get_queue_url(queue_name, sqs_client)
    sqs_client.delete_message(QueueUrl=queue_url, ReceiptHandle=receipt_handle)


def remove_messages(queue_name: str, receipt_handles: List[str], sqs_client=None):
    queue_url, sqs_client = get_queue_url(queue_name, sqs_client)
    # TODO implement batch version
    for receipt_handle in receipt_handles:
        sqs_client.delete_message(QueueUrl=queue_url, ReceiptHandle=receipt_handle)


def return_message(queue_name: str, receipt_handle: str, sqs_client=None):
    queue_url, sqs_client = get_queue_url(queue_name, sqs_client)
    sqs_client.change_message_visibility(QueueUrl=queue_url, ReceiptHandle=receipt_handle, VisibilityTimeout=0)


def return_messages(queue_name: str, receipt_handles: List[str], sqs_client=None):
    queue_url, sqs_client = get_queue_url(queue_name, sqs_client)
    # TODO implement batch version
    for receipt_handle in receipt_handles:
        sqs_client.return_message(QueueUrl=queue_url, ReceiptHandle=receipt_handle)

def send_messages(queue_name: str, messages: list, group_id: Optional[str] = None, sqs_client=None) -> list:
    """
    Returns: list os failed ids
    """
    queue_url, sqs_client  = get_queue_url(queue_name, sqs_client)
    failed_ids = []
    messages_list = messages.copy()
    ids = list(range(len(messages_list)))
    while len(messages_list) > 0:
        # prepare a batch of up to 10 messages
        batch_items = []
        while len(messages_list) > 0 and len(batch_items) < 10:
            item = {'Id': str(ids.pop(0)), 'MessageBody': messages_list.pop(0)}
            if group_id is not None:
                item['MessageGroupId'] = group_id
            batch_items.append(item)
        result = sqs_client.send_message_batch(QueueUrl=queue_url, Entries=batch_items)
        if result.get('Failed'):
            failed_ids.extend([int(item['Id']) for item in result.get('Failed')])
    failed_messages = [messages[i] for i in failed_ids]
    return failed_messages


def receive_messages(queue_name, max_messages=10, wait_seconds=0, sqs_client=None) -> Dict[(str, str)]:
    """Returns a dict of {message_handle: body}"""
    queue_url, sqs_client  = get_queue_url(queue_name, sqs_client)

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
            # ku.log_trace(f'sqs: {len(new_results)} messages read from {queue_name}')
        else:
            # ku.log_trace(f'sqs: 0 messages read from {queue_name}')
            break
    return result_messages


def queue_attributes(queue_name, sqs_client=None) -> dict:
    queue_url, sqs_client  = get_queue_url(queue_name, sqs_client)
    result = sqs_client.get_queue_attributes(QueueUrl=queue_url, AttributeNames=['All'])
    return result['Attributes']


def purge_queue(queue_name, sqs_client=None):
    queue_url, sqs_client  = get_queue_url(queue_name, sqs_client)
    result = sqs_client.purge_queue(QueueUrl=queue_url)