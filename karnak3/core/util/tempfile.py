import os
import tempfile
import shutil
import glob
import datetime
import uuid
from typing import Optional

import karnak3.core.log as kl
import karnak3.core.util.path as kp


def temp_file_standard_prefix() -> str:
    prefix = 'karnak.'
    # if KarnakConfig.module_name is not None:
    #     prefix += KarnakConfig.module_name + '.'
    return prefix


def temp_file_prefix(prefix: Optional[str] = None) -> str:
    std_prefix = temp_file_standard_prefix()
    if prefix is None:
        prefix = std_prefix
    elif not prefix.startswith(std_prefix):
        prefix = std_prefix + prefix
    return prefix


def create_temp_file(prefix=None, suffix='.tmp', binary: bool = False, delete: bool = False):
    new_prefix = temp_file_prefix(prefix)
    mode = 'w+'
    if binary:
        mode = 'w+b'
    f = tempfile.NamedTemporaryFile(mode=mode, delete=delete, prefix=new_prefix, suffix=suffix)
    return f


# def create_temp_dir(prefix=None) -> str:
#     new_prefix = temp_file_prefix(prefix)
#     # path = str_ensure_prefix(tempfile.mkdtemp(prefix=prefix))
#     path = tempfile.mkdtemp(prefix=new_prefix)
#     return path


def cleanup_temp_files():
    tmp_folder = tempfile.gettempdir()
    std_prefix = temp_file_standard_prefix()
    glob_files = '{}/{}*'.format(tmp_folder, std_prefix)
    glob_dirs = glob_files + '/'
    assert glob_files.find('karnak.') >= 0
    assert glob_dirs.find('karnak.') >= 0

    tmp_dirs = glob.glob(glob_dirs)
    for d in tmp_dirs:
        kl.debug("removed temp folder: {}", d)
        assert d.find('karnak.') >= 0
        shutil.rmtree(d)

    tmp_files = glob.glob(glob_files)
    for i in tmp_files:
        try:
            assert i.find('karnak.') >= 0
            os.remove(i)
            kl.debug("removed temp file: {}", i)
        except OSError:
            kl.warn("error while deleting temp file: {}", i)


def unique_temp_filename(prefix: Optional[str] = None, suffix: str = '.tmp') -> str:
    _prefix = temp_file_prefix(prefix)
    return (_prefix + datetime.datetime.now().strftime("%y%m%d_%H%M%S")
            + '_' + uuid.uuid4().hex + suffix)


def unique_temp_filepath(prefix: Optional[str] = None, suffix: str = '.tmp') -> str:
    tmp_folder = tempfile.gettempdir()
    filename = unique_temp_filename(prefix=prefix, suffix=suffix)
    filepath = kp.path_concat(tmp_folder, filename)
    return filepath
