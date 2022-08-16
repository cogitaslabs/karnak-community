import os
import tempfile
import shutil
from typing import Optional, Type, Dict
from urllib.request import pathname2url
from urllib.parse import urljoin

import pandas as pd

import karnak3.core.util as ku
import karnak3.core.store as kcs
from karnak3.core.store import KStoreEngine, KStoreConfig, KStoreObject


class KStoreFileConfig(KStoreConfig):
    def __init__(self, temp_filepath: Optional[str] = None):
        self.temp_filepath: str = ku.coalesce(temp_filepath, tempfile.tempdir)


class KStoreFileObject(KStoreObject):
    def __init__(self, url: str, # engine: Optional['KStoreFileEngine'] = None,
                 config: Optional[KStoreFileConfig] = None):
        super().__init__(resource_type='localfile-object', url=url)
        # self.engine: Optional['KStoreFileEngine'] = engine
        # if engine is None:
        #     self.engine: Optional['KStoreFileEngine'] = kcs.get_engine(url)
        self.config = ku.coalesce(config, KStoreFileConfig())

    @classmethod
    def is_localfile(cls) -> bool:
        return True

    def object_from_filepath(self, filepath: str) -> 'KStoreFileObject':
        url = urljoin('file:', pathname2url(filepath))
        return KStoreFileObject(url, # engine=self.engine,
                                config=self.config)

    def tempfile_object(self, suffix: Optional[str] = None) -> 'KStoreObject':
        return self.object_from_filepath(self.temp_filepath(suffix=suffix))

    def save_file(self, filepath: str) -> 'KStoreFileObject':
        # copy file if origin !+ destination
        if not os.path.samefile(self.path, filepath):
            shutil.copyfile(self.path, filepath)
        return self.object_from_filepath(filepath)

    def load_file(self, filepath: Optional[str]) -> 'KStoreFileObject':
        if filepath is None:
            target = self.tempfile_object()
        else:
            target = self.object_from_filepath(filepath)
        return target.save_file(self.path)


class KStoreFileEngine(KStoreEngine):
    def __init__(self):
        super().__init__('file')

    def build_store_object(self, url: str,
                           config: Optional[KStoreFileConfig] = None) -> KStoreFileObject:
        return KStoreFileObject(url,  # engine=self,
                                config=config)

    def save_file(self, url: str, filepath: str,
                  config: Optional[KStoreFileConfig] = None) -> KStoreFileObject:
        return self.build_store_object(url, config=config).save_file(filepath)

    def load_file(self, url: str, filepath: Optional[str],
                  config: Optional[KStoreConfig] = None) -> KStoreFileObject:
        return self.build_store_object(url, config=config).load_file(filepath)


# register engine
_engine = KStoreFileEngine()
