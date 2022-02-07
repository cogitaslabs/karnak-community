from typing import List, Optional

import karnak3.core.log as kl
import karnak3.cloud.aws as kca


def sts_credentials(duration_seconds=3600*12, aws_region: Optional[str] = None):
    sts_client = kca.get_aws_client('sts', aws_region=aws_region)
    cred = sts_client.get_session_token(DurationSeconds=duration_seconds)['Credentials']
    caller_identity = sts_client.get_caller_identity()
    kl.trace('sts caller identity: {}', str(caller_identity))
    return cred['AccessKeyId'], cred['SecretAccessKey'], cred['SessionToken'], cred['Expiration']

