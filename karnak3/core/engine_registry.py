from abc import abstractmethod
from typing import Dict, Optional, List, Any
from urllib.parse import urlparse

import karnak3.core.util as ku
import karnak3.core.log as kl


class KarnakRegistryException(ku.KarnakException):
    pass


class KarnakResource:
    def __init__(self, resource_type: str, resource_path: str):
        self.resource_type = resource_type
        self.resource_path = resource_path

    def get_content(self) -> Any:
        raise KarnakRegistryException(f'resource {self.resource_type} '
                                      f'does not support method get_content')


class KarnakEngine:

    def __init__(self, key: str):
        self.key = key

    # def key(self) -> str:
    #     return self.key

    def start(self, force: bool = False):
        pass

    def is_valid_resource_name(self, resource_name: str) -> bool:
        return resource_name == self.key

    def get_resource(self, resource_name) -> KarnakResource:
        raise KarnakRegistryException(f'engine {self.key} '
                                      f'does not support method choose_resource')


class KarnakRegistry:

    def __init__(self, name: str):
        self.name = name
        self._engines: Dict[str, KarnakEngine] = {}

    def register(self, engine: KarnakEngine):
        key = engine.key
        if key not in self._engines:
            self._engines[key] = engine
            kl.debug("engine registered for '{}' at {}", key, self.name)
            kl.debug("registered engines at {}: {}", self.name, ', '.join(self._engines.keys()))
        else:
            kl.debug("engine already registered for '{} at {}'", key, self.name)

    def start_engines(self, keys: Optional[List[str]], force: bool = False):
        _keys = keys if keys is not None else self._engines.keys()
        for k in keys:
            self._engines[k].start(force=force)

    def registered_keys(self) -> List[str]:
        return list(self._engines.keys())

    def get_engine_by_key(self, key: str) -> Optional[KarnakEngine]:
        return self._engines.get(key)

    def get_engine(self, resource_name: str,
                   include_missing: bool = True) -> Optional[KarnakEngine]:
        for k in self._engines.keys():
            engine = self._engines[k]
            if engine.is_valid_resource_name(resource_name):
                return engine
        if include_missing:
            return self.get_missing_engine(resource_name)
        else:
            return None

    def get_missing_engine(self, resource: str) -> Optional[KarnakEngine]:
        raise KarnakRegistryException(f'no engine found in registry {self.name}'
                                      f' for resource {resource}')

    def get_resource(self, resource_name: str,
                     include_missing: bool = True) -> Optional[KarnakResource]:
        engine = self.get_engine(resource_name, include_missing)
        if not engine:
            return None
        return engine.get_resource(resource_name)

    def get_resource_content(self, resource_name: str,
                             include_missing: bool = True) -> Any:
        resource = self.get_resource(resource_name, include_missing)
        if not resource:
            return None
        return resource.get_content()


#
# URI Engine
#

class KUriRegistryException(KarnakRegistryException):
    pass


class KarnakUriResource(KarnakResource):

    def __init__(self, resource_type: str, uri: str):
        super().__init__(resource_type=resource_type, resource_path=uri)
        self.uri = uri
        parsed_uri = urlparse(uri)
        self.scheme = parsed_uri.scheme
        self.netloc = parsed_uri.netloc
        self.hostname = parsed_uri.hostname
        self.port = parsed_uri.port
        self.path = parsed_uri.path
        self.params = parsed_uri.params
        self.query = parsed_uri.query
        self.fragment = parsed_uri.fragment


class KarnakUriEngine(KarnakEngine):

    def __init__(self, scheme: str):
        super().__init__(key=scheme)
        self.scheme = scheme

    # @abstractmethod
    # def name(self) -> str:
    #     return ''

    # def _protocol_prefix(self) -> str:
    #     return self.protocol + '://'

    def is_valid_resource_name(self, resource_name: str) -> bool:
        parsed_uri = urlparse(resource_name)
        return parsed_uri.scheme == self.scheme

    def is_valid_uri(self, uri: str) -> bool:
        return self.is_valid_resource_name(uri)

    # def resource_path(self, uri: str):
    #     # TODO use uri parser
    #     return uri[len(self._protocol_prefix()):] if self.is_valid_uri(uri) else None

    # def build_uri(self, path_start, *more_path):
    #     path = path_concat(path_start, *more_path)
    #     return self.protocol_prefix() + path

    # # TODO review
    # def build_uri_from_host_path(self, host, path):
    #     return self._protocol_prefix() + host + ku.str_ensure_prefix(path, '/')


# class KUriRegistry(KarnakRegistry):
#
#     @staticmethod
#     def protocol(uri: str) -> str:
#         p = uri.split('://')[0]
#         if p is None or len(p) == 0:
#             raise KUriRegistryException('invalid uri: {}'.format(uri))
#         return p
#
#     @classmethod
#     def parse_uri(cls, uri) -> (str, str):
#         """return protocol, host, path (currently, not complete URI parse)"""
#         protocol, remainder = uri.split('://', 1)
#         host, path = remainder.split('/', 1)
#         return protocol, host, '/' + path   # path should start with /
