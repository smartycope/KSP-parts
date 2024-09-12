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
import streamlit as st
from default_config import default_config
from utils import *
from load_data import *

st.set_page_config(page_title='KSP part database', page_icon=None, layout="wide", initial_sidebar_state="expanded")
ss = st.session_state

if 'all_categories' not in ss:
    ss['all_categories'] = ['Engine', 'Generate an initial dataset to load available fields']

if 'all_providers' not in ss:
    ss['all_providers'] = ['Generate an initial dataset to load available fields']

engine_column_order = [
    'icon_url',
    'title',
    'TWR',
    'mass',
    'cost',
    'maxThrust',
    'ISP_atm',
    'ISP_vac',
    'ISP_atm_WR',
    'ISP_vac_WR',
    'techLevel',
    'gimbalRange',
    'alternatorRate',
    'crashTolerance',
    'maxTemp',
    'heatConductivity',
    'provider',
    'manufacturer',
    'name',
]

@st.cache_data
def _load_localization(localization_paths):
    return load_localization(localization_paths)

@st.cache_data
def _load_parts(paths):
    try:
        results = load_parts(paths)
    except:
        'Error: Try reloading the page'

    ss['parts_not_loaded'] = []
    ss['parts_loaded'] = []
    for i in results:
        if type(i) is str:
            ss['parts_not_loaded'].append(i)
        else:
            ss['parts_loaded'] += i
    return ss['parts_loaded'], ss['parts_not_loaded']

@st.cache_data
def _load_thumbs(thumbs_paths, PATH_GAMEDATA_OFFSET):
    return load_thumbs(thumbs_paths, PATH_GAMEDATA_OFFSET)

@st.cache_resource
def start_icon_server(port, dir):
    class MyHandler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=dir, **kwargs)

    def start_server():
        with socketserver.TCPServer(("", port), MyHandler) as httpd:
            print(f"Serving at port {port}")
            httpd.serve_forever()

    # Run the server in a separate thread
    server_thread = threading.Thread(target=start_server)
    server_thread.daemon = True
    server_thread.start()

    print("Icon server started in a separate thread")


with st.sidebar:
    LANG = st.text_input('Language ID', 'en-us', 5)
    INSTALLATION_PATH = Path(st.text_input('Installation Path', '/home/zeke/Software/KSP_linux'))

    # Set the paths from the root path
    part_db_path, part_dump_file, loc_dump_file, localization_paths, thumbs_paths = construct_paths(INSTALLATION_PATH, LANG)

    # Manually invalidate the caches
    if st.button('Reload Parts'):
        _load_parts.clear()
    if st.button('Reload Thumbnails'):
        _load_thumbs.clear()
    if st.button('Reload Localization Strings'):
        _load_localization.clear()

    with st.expander('Configuration'):
        """
            ### Specify how to construct the dataset
            (This is easiest to edit if you make it full screen)
            - The `JSONPath` column follows [this](https://pypi.org/project/jsonpath-ng/) JSONPath syntax
            - To try multiple paths, you can put ` -> ` in between each JSONPath. It will try each one, in order,
                until it finds a valid value
            - JSONPath *must* start with $. (why wouldn\'t it?)
            - The structure is essentially the same as the .cfg file structure (the file path for each part is in the
                `path` column, if you want to see the structure), with a few modifications. If there are multiple keys
                in a given scope, it puts all the values in a list. Same with the scope names.
            - If a valid polars expression is given, it will apply the calculation. Calculations are performed in order
                (of the rows), and c = pl.col
            - The row order of this defines the column order of the dataset
            - The `icon_url`, `icon_path`, `provider`, and `path` columns are automatically included and not specified here
            - Types are ignored for calculated columns
            - Compound types are supported (i.e. list[int])
            - You can specify a column twice, once to get the data, and again later to clean the data.
                - Note that if you do, it won't get cast automatically, you'll have to cast it yourself manually in the
                cleaning code
            - For casting, if you put a `*` in front of the type, that indicates that the JSONPath fetches a list of values,
                instead of a single value, and should be treated as a list. If you just do `list[type]`, it will attempt
                to parse it as a list (delimited either by `,` or ` `)
        """
        fields, calculated, localized, types, default_fields, column_order, natively_list = \
            construct_config(
                st.data_editor(pd.DataFrame(default_config, columns=
                    ['Column Name', 'JSONPath', 'Localized?', 'Data Type', 'Included by Default?']
                ))
            )

# Collect the paths from the part database
paths = get_paths(part_db_path)

PATH_GAMEDATA_OFFSET = len(str(INSTALLATION_PATH / 'GameData'))

string_lookup = _load_localization(localization_paths)
all_parts, failed = _load_parts(paths)
thumbs = _load_thumbs(thumbs_paths, PATH_GAMEDATA_OFFSET)
start_icon_server(PORT, INSTALLATION_PATH / 'GameData')

f"{len(ss['parts_loaded'])}/{len(ss['parts_loaded'])+len(ss['parts_not_loaded'])} parts loaded. {len(ss['parts_not_loaded'])} parts failed to parse"
if len(ss['parts_not_loaded']) > 0:
    st.json(ss['parts_not_loaded'], expanded=False)

# Generate the dataset
with st.form('generate dataset'):
    selected_fields = ['icon_url', 'provider'] + list(set(st.multiselect('Feilds', list(fields) + list(calculated), default_fields)))

    @st.cache_data
    def _construct_df():
        print('Regenerating Dataset')
        df = construct_df(fields, all_parts, localized, string_lookup, types, calculated, natively_list, thumbs, PATH_GAMEDATA_OFFSET, selected_fields)
        ss['all_categories'] = df['category'].unique()
        ss['all_providers'] = df['provider'].unique()
        return df

    df = _construct_df()

    if 'category' in df.columns:
        categories = st.multiselect('Filter by Part Category', sorted(st.session_state['all_categories']), ['Engine'])
        filtered = df.filter(c('category').is_in(categories))

    if 'provider' in df.columns:
        mods = st.multiselect('Filter by Provider (mod)', sorted(st.session_state['all_providers']), sorted(st.session_state['all_providers']))
        filtered = filtered.filter(c('provider').is_in(mods))

    st.form_submit_button('Generate Dataset', on_click=_construct_df.clear)

# Show the dataset
st.dataframe(filtered.to_pandas(),
    hide_index=True,
    column_config={
        'icon_url': st.column_config.ImageColumn('Icon')
    },
    # icon_url SHOULD ALREADY BE IN column_order
    # WHAT THE CRAP
    column_order=['icon_url'] + column_order,
)

# Searching
with st.expander('Search for Part', False):
    try:
        query = st.text_input('Search')
        use_regex = st.checkbox('Use Regex', True)
        if use_regex:
            use_case = st.checkbox('Case Sensistive')
        search_all2 = st.checkbox('Search *All* Parts', True)

        query = ('(?i)' if use_regex and not use_case else '') + query

        searched = ((df if search_all2 else filtered).filter(
            c('title').str.contains(query, literal=not use_regex, strict=False) |
            c('name').str.contains(query, literal=not use_regex, strict=False) |
            c('description').str.contains(query, literal=not use_regex, strict=False)
        ))
        st.dataframe(searched.to_pandas(),
            hide_index=False,
            column_config={
                'icon_url': st.column_config.ImageColumn('Icon')
            },
            column_order=column_order,
        )
    except Exception as err:
        print(f'ERROR: Line {__line__()}:', err)
        "Please generate a dataset first"

# Select Engine
with st.expander('Select Engine', False):
    try:
        # ['supersonicFlight', 'start', 'heavyLanding', 'basicScience', 'advLanding', 'fuelSystems', 'highAltitudeFlight', 'electrics', 'largeVolumeContainment', 'advMetalworks', 'stability', 'advConstruction', 'basicRocketry', 'ionPropulsion', 'automation', 'largeElectrics', 'generalConstruction', 'nuclearPropulsion', 'electronics', 'advAerodynamics', 'precisionEngineering', 'engineering101', 'aerospaceTech', 'advFlightControl', 'landing', 'specializedConstruction', 'veryHeavyRocketry', 'advancedMotors', 'experimentalAerodynamics', 'advElectrics', 'advRocketry', 'experimentalElectrics', 'generalRocketry', 'aviation', 'hypersonicFlight', 'precisionPropulsion', 'specializedControl', 'actuators', 'commandModules', 'heavyRocketry', 'metaMaterials', 'propulsionSystems', 'scienceTech', None, 'advUnmanned', 'largeUnmanned', 'heavierRocketry', 'survivability', 'experimentalScience', 'composites', 'aerodynamicSystems', 'unmannedTech', 'advScienceTech', 'advFuelSystems', 'advExploration', 'unavailable', 'spaceExploration', 'Unresearcheable', 'nanolathing', 'Unresearchable', 'highPerformanceFuelSystems', 'specializedElectrics', 'fieldScience', 'miniaturization', 'heavyAerodynamics', 'flightControl']
        search_all = st.checkbox('Search *All* Parts', True, key=1)
        engines = (df if search_all else filtered).filter(c('category') == 'Engine')
        # tech_levels = st.multiselect('Tech Levels Allowed', engines['TechRequired'].unique(), engines['TechRequired'].unique())
        min_thrust = st.slider('Min Thrust', engines['maxThrust'].min(), engines['maxThrust'].max())
        gimbal = st.slider('Minumum Gimbal Range', engines['gimbalRange'].min(), engines['gimbalRange'].max())
        atm = st.checkbox('In Atmosphere')

        pand2 = (engines
            .filter(
                # c('TechRequired').
                c('maxThrust') >= min_thrust,
                # Not quite sure how this works
                c('gimbalRange') >= gimbal,
                c('ISP_atm' if atm else 'ISP_vac') > 0,
            )
            .sort('ISP_atm_WR' if atm else 'ISP_vac_WR')
            .select(engine_column_order)
            .unique()
        ).to_pandas()
        st.dataframe(pand2,
            hide_index=True,
            column_config={
                'icon_url': st.column_config.ImageColumn('Icon')
            },
            column_order=engine_column_order,
        )
    except Exception as err:
        print(f'ERROR: Line {__line__()}:', err)
        # raise err
        "Error: Please generate a dataset first"
