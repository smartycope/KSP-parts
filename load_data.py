from utils import parse_cfg_file, process_file, PORT
from pathlib import Path
import polars as pl
import pandas as pd
import http.server
import socketserver
import threading
import re
import ezregex as er
from jsonpath_ng.ext import *
from pyparsing import *
from builtins import filter
import itertools as it
import multiprocessing
from polars import col as c

def load_localization(localization_paths):
    raw_strings = []
    for i in localization_paths:
        try:
            raw_strings += parse_cfg_file(i).as_list()[0][1][1:]
        except ParseBaseException as err:
            print(f'Failed to parse {i}: {err}')

    return (pl.DataFrame(raw_strings, orient='row')
        .select(id='column_0', string=c('column_1').str.strip_chars(' '))
    )

def load_parts(paths):
    with multiprocessing.Pool() as pool:
        results = pool.map(process_file, paths)
    return results

def load_thumbs(thumbs_paths, PATH_GAMEDATA_OFFSET):
    thumbs = []
    for k in list(it.chain(*[list(i.iterdir()) for i in thumbs_paths])):
        try:
            thumbs.append((str(k), re.match(r'(.+)_icon0?\D.+', k.name).group(1).replace('.', '_')))
        except Exception as err:
            # print(k)
            continue
    return pl.DataFrame(thumbs, orient='row').select(path='column_0', part='column_1', url=f'http://0.0.0.0:{PORT}' + c('column_0').str.slice(PATH_GAMEDATA_OFFSET,))
