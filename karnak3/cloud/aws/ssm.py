from typing import List, Optional

import karnak3.core.log as kl
import karnak3.cloud.aws as kca
from core.engine_registry import KarnakUrlEngine, KarnakUrlResource


def ssm_get_parameter(name, aws_region: Optional[str] = None) -> str:
    ssm_client = kca.get_aws_client('ssm', aws_region=aws_region)
    parameter_dict = ssm_client.get_parameter(Name=name, WithDecryption=True)
    param_value = parameter_dict['Parameter']['Value']
    return param_value


def ssm_get_parameters_by_path(path: str, aws_region: Optional[str] = None) -> List[str]:
    ssm_client = kca.get_aws_client('ssm', aws_region=aws_region)
    kl.trace('getting ssm path:{}', path)
    parameter_dict = ssm_client.get_parameters_by_path(Path=path, WithDecryption=True,
                                                       Recursive=True)
    params = parameter_dict['Parameters']
    param_value_list = [p['Value'] for p in params]
    return param_value_list


class SsmStoreResource(KarnakUrlResource):

    def get_content(self) -> str:
        content = ssm_get_parameter(self.resource_uri)
        return content


class SsmStoreEngine(KarnakUrlEngine):
    def __init__(self):
        super().__init__('ssm')

    def start(self, force: bool = False):
        kca.get_aws_client('ssm')


_ssm_store_engine = SsmStoreEngine()
