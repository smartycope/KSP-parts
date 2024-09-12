import math
import numpy as np
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
import save_file_part_templates as templates
from operator import add, sub

PORT = 8000
diameters = ('size0', 'size1', 'size1p5', 'size2', 'size3', 'size4')
arbitrary_id = 1
NULL_PART = 'Null_0'
DEBUG = False

# Unknown
'FL-A151L Fuel Tank Adapater'
'Kerbodyne ADTP 2-3'
adapters = {
    ('size0', 'size1'): 'FL-A10 Adapter',
    ('size1', 'size2'): 'Rockomax Brand Adapter',
    # ('size1', 'size2'): 'C7 Brand Adapter - 2.5m to 1.25m',
    ('size2', 'size3'): 'Kerbodyne ADTP-2-3A',
    ('size3', 'size4'): 'Kerbodyne S3-S4 Adapter Tank',

    ('size0', 'size1p5'): 'FL-A150 Fuel Tank Adapter',
    ('size1', 'size1p5'): 'FL-A151S Fuel Tank Adapter',
    ('size1p5', 'size2'): 'FL-A215 Fuel Tank Adapter',
}


def __line__():
    from inspect import currentframe
    cf = currentframe()
    return cf.f_back.f_lineno

def istype(obj:type, typ:type|tuple[type]):
    if not isinstance(typ, (list, tuple, set)):
        typ = (typ,)
    return obj in typ or (hasattr(obj, '__origin__') and obj.__origin__ in typ)

assert istype(list[int], list)
assert istype(list[float], list)
assert istype(list[tuple[int]], list)
assert istype(list, list)
assert not istype(tuple, list)
assert not istype(tuple[int], list)
assert not istype(tuple[list], list)
assert not istype(list, list[int])
assert istype(list, (list, int))
assert istype(list[float], (list, int))
assert not istype(list[float], (float, int))

def convert_list_to_dict(data):
    def parse_block(block):
        parsed = {}
        for item in block:
            if isinstance(item, list) and len(item) > 1 and isinstance(item[1], list):
                key = item[0]
                value = parse_block(item[1:])
                if key in parsed:
                    if isinstance(parsed[key], list):
                        parsed[key].append(value)
                    else:
                        parsed[key] = [parsed[key], value]
                else:
                    parsed[key] = value
            elif isinstance(item, list) and len(item) == 2:
                key, value = item
                value = value.strip()
                if key in parsed:
                    if isinstance(parsed[key], list):
                        parsed[key].append(value)
                    else:
                        if parsed[key] != value:
                            parsed[key] = [parsed[key], value]
                else:
                    parsed[key] = value
        return parsed

    result = {}
    for item in data:
        if isinstance(item, list) and len(item) > 1:
            key = item[0]
            value = parse_block(item[1:])
            if key in result:
                if isinstance(result[key], list):
                    result[key].append(value)
                else:
                    result[key] = [result[key], value]
            else:
                result[key] = [value]
    return result

def remove_bom(data):
    BOM = b'\ufeff'.decode('utf-8')
    if data.startswith(BOM):
        return data[len(BOM):]
    return data

def parse_cfg_file(file_path, verbose=False):
    # The parser for the, as far as I can tell, custom .cfg files
    # This ignores the requirement clause
    identifier = Word(alphanums + "_-!#%") + Opt(Suppress(':'+ Word(alphanums + "_-!#%:[]")))

    # requirement = LineStart() + Literal('@') + Literal('{') + ... + Literal('}') # nested_expr('{', '}', empty) #+ CharsNotIn('}') + '}' # + nested_expr('{', '}', empty)
    # nested = Forward()
    # nested_content = nested_expr('{', '}', ZeroOrMore(nested | empty))
    # nested <<= nested_expr('{', '}', ZeroOrMore(nested))
    # anything = CharsNotIn('@')
    requirement = LineStart() + '@' + ... + StringEnd()#+ anything + nested_expr('{', '}', anything)
    # requirement = LineStart() + "@" + identifier + nestedExpr("{", "}")

    # Define grammar for key-value pairs
    key_value = Group(identifier + Suppress('=') + rest_of_line)# + CharsNotIn('}')

    # Create a placeholder for the block
    block = Forward()

    # Define the content of the block
    content = ZeroOrMore(block | key_value)

    # Define the block using the placeholder
    # block <<= Group(identifier + nested_expr('{', '}', content))
    block <<= Group(identifier + Suppress('{') + content + Suppress('}'))
    # cfg_grammar = OneOrMore(block)
    cfg_grammar = ZeroOrMore(key_value) + OneOrMore(block)

    if verbose: print(f'Starting to parse {file_path}')
    with open(file_path, 'r', encoding='utf-8-sig') as file:
        if verbose: print(f'Reading {file_path}...', end='')
        content = file.read()
        if verbose: print(f'Done Reading')

    # Remove BOM if present
    if verbose: print(f'Removing BOM...', end='')
    content = remove_bom(content)
    if verbose: print(f'Done Removing BOM')
    # Remove comments
    if verbose: print(f'Removing Comments...', end='')
    content = dblSlashComment.suppress().transformString(content)
    if verbose: print(f'Done Removing Comments')
    # Remove the requirements
    # if verbose: print(content)
    if verbose: print(f'Removing Requirements...', end='')
    content = requirement.suppress().transformString(content)
    if verbose: print(f'Done Removing Requirements')
    if verbose: print(content)
    # content = content.replace('\n\n', '\n')
    # Parse the content
    if verbose: print(f'Parsing Grammar...', end='')
    parsed_data = cfg_grammar.parseString(content, parseAll=True)
    if verbose: print(f'Done Parsing Grammar')

    return parsed_data

def process_file(path):
    try:
        # print(f'Started converting {path}')
        rtn = convert_list_to_dict(parse_cfg_file(path).as_list())['PART']
        for i in rtn:
            i['path'] = path
        # print(f'Finished converting {path}')
        return rtn
    except Exception as err:
        print(f'Error parsing {path}:\n{err}')
        return path
        # parse_cfg_file(path, True)
        # raise Exception(f'Error parsing {path}:\n{err}')

def construct_config(config):
    fields = {}
    calculated = {}
    localized = []
    types = {}
    default_fields = []
    natively_list = []
    for idx, i in config.iterrows():
        col, path, loc, typ, default = i
        if path.startswith('$.'):
            fields[col] = path.split('->') if '->' in path else path
        else:
            try:
                calculated[col] = [eval(i, globals(), locals()) for i in (path.split('->') if '->' in path else [path])]
            except Exception as err:
                print(f'Failed to parse JSONPath {path}:', err)
        try:
            if typ.startswith('*'):
                natively_list.append(col)
                typ = typ[1:]
            types[col] = eval(typ)
        except Exception as err:
                print(f'Failed to parse type {path}:', err)
        if loc:
            localized.append(col)
        if default:
            default_fields.append(col)
    column_order = ['icon_url' + 'provider'] + list(fields.keys())

    return fields, calculated, localized, types, default_fields, column_order, natively_list

def get_paths(part_db_path):
    with part_db_path.open() as f:
        raw_paths = re.finditer(str('url = ' + er.group(er.chunk)), f.read())
    paths = [
        str((part_db_path.parent / 'GameData' / path.group(1)).parent) + '.cfg'
        for path in raw_paths
    ]
    return paths

def construct_paths(INSTALLATION_PATH, LANG='en-us'):
    part_db_path = INSTALLATION_PATH / 'PartDatabase.cfg'
    part_dump_file = Path.home() / 'Documents' / 'KSP_parts_dump.json'
    loc_dump_file = Path.home() / 'Documents' / 'KSP_localization_dump.json'
    localization_paths = [
        INSTALLATION_PATH / 'GameData/Squad/Localization/dictionary.cfg',
        INSTALLATION_PATH / 'GameData/SquadExpansion/Serenity/Localization/dictionary.cfg',
        *(INSTALLATION_PATH / 'GameData').rglob(f'{LANG}.cfg')
    ]
    thumbs_paths = list((INSTALLATION_PATH / 'GameData').rglob('@thumbs'))

    return part_db_path, part_dump_file, loc_dump_file, localization_paths, thumbs_paths

def construct_df(fields, all_parts, localized, string_lookup, types, calculated, natively_list, thumbs, PATH_GAMEDATA_OFFSET, selected_fields=None):
    # Construct the dataset using the specified fields
    parsed_fields = {
        col: parse(path) if type(path) is str else [parse(p) for p in path]
        for col, path in fields.items()
        # if col in selected_fields
    }
    df_data = {col: [] for col in fields.keys()}
    for i in all_parts:
        for col, field in parsed_fields.items():
            # If it's just a string, get it
            if not isinstance(field, (list, tuple)):
                l = [m.value for m in field.find(i)]
            else:
                # Otherwise, it's an iterable. Keep trying them in order until you get something
                for k in field:
                    try:
                        l = [m.value for m in k.find(i)]
                    except KeyError:
                        continue
                    if len(l) > 0:
                        break

            # It didn't get anything
            if len(l) == 0:
                l = [None]
            # We got a list of things, ensure it stays a list. Or we got a single thing, but it's *supposed* to be a list
            elif len(l) > 1 or (col in natively_list):
                l = [l]
            df_data[col] += l

    # Fine tune the constructed dataset
    df = pl.DataFrame(df_data)

    # Localize columns
    for col in localized:
        # if col in selected_fields:
        df = (df.join(string_lookup, how='left', right_on='id', left_on=col)
            .with_columns(
                pl.when(c('string').is_null())
                .then(col)
                .otherwise('string')
                .alias(col)
            ).drop('string')
        )

    # Cast columns
    for col, typ in types.items():
        # if col not in selected_fields or col in calculated:
        if col in calculated:
            continue
        if typ is bool:
            # It's already a list, just cast all the elements of all of them
            if col in natively_list:
                df = df.with_columns(c(col).list.eval(pl.element().str.to_lowercase().is_in(('true', '1'))))
            else:
                df = df.with_columns(c(col).str.to_lowercase().is_in(('true', '1')))
        elif istype(typ, (list, tuple)):
            # If it's already a list, perfect, we don't need to do anything
            # if df[col].dtype.base_type() is pl.List: continue
            # TODO: I don't feel like doing element-wise list casting right now
            if col in natively_list: NotImplemented

            examples = df[col].drop_nulls()
            # If they're all null, we can't do anything
            if not len(examples): continue

            comma = ',' in examples[0]
            df = df.with_columns(c(col).str.split(',' if comma else ' '))
            if hasattr(typ, '__args__'):
                try:
                    df = df.with_columns(c(col).list.eval(pl.element().str.strip_chars(' ,').cast(typ.__args__[0])))
                except Exception as err:
                    raise Exception(f'Failed to cast column `{col}` to `{typ}`: {err}')
        else:
            try:
                if col in natively_list:
                    df = df.with_columns(c(col).list.eval(pl.element().cast(typ)))
                else:
                    df = df.with_columns(c(col).cast(typ))
            except Exception as err:
                raise Exception(f'Cannot cast column {col} to {typ}: {err}') from err

    # Calculate the additional columns
    for col, calc in calculated.items():
        if type(calc) is not list:
            # Run them one at a time so they can depend on each other
            try:
                df = df.with_columns(calc.alias(col))
            except Exception as err:
                raise Exception(f'Error with custom calculation `{calc}` for column `{col}`: {err}') from err
        else:
            # Same thing, but try each one to see which works first
            success = False
            error = None
            for i in calc:
                try:
                    df = df.with_columns(i.alias(col))
                    success = True
                    break
                except Exception as err:
                    error = err
                    continue
            if not success:
                raise Exception(f'Error with custom calculation `{calc}` for column `{col}`: {error}') from error

    df = (df
        # Add the icons
        .join(thumbs, how='left', left_on='name', right_on='part').rename({'path_right': 'icon_path', 'url': 'icon_url'})
        .with_columns(
            provider=c('path').str.slice(PATH_GAMEDATA_OFFSET,).str.split('/').list[1],
            # node_stack_top=c('node_stack_top').fill_null([0, 0, 0, 0, 1, 0, 1]),
            # node_stack_bottom=c('node_stack_bottom').fill_null([0, 0, 0, 0, 1, 0, 1]),
        )
    )

    # The "Schnauser" engine doesn't seem to exist in-game?
    df = df.filter(c('name') != 'restock-engine-schnauzer-1')

    if selected_fields is not None:
        df = df.select(selected_fields)

    return df.unique('name')
    # return rtn.join(rtn.group_by('name').agg(pl.first()), 'name')
    # assert len(set(df['name'])) == len(df['name']), 'Non-unique names detected!'
    # Ensure parts are uniuqe by name
    # return


def deltaV(isp, initial_mass, final_mass, g=9.81) -> float:
    # NOTE: The .8 is a fudge factor. Right now, it keeps calculating ISP too low according to the game
    return math.log(initial_mass / final_mass) * isp * g * .8

def optimal_engine(parts:pl.DataFrame, dry_mass, wet_mass, diameter, vac:bool, min_TWR: float, propellent=['LiquidFuel', 'Oxidizer']) -> pl.DataFrame:
    """ Make TWR <=0 to ignore it """
    return (parts
        .filter(
            (c('diameter').list.set_intersection(diameters[:diameters.index(diameter)+1]).list.len() > 0) &
            (c('maxThrust')/(wet_mass + c('wet_mass')) >= min_TWR*10) & # apparently TWR is in different units
            (c('propellants') == propellent) &
            # Make sure it's not radially mounted
            (~c('node_stack_top').is_null())
            # (~c('node_stack_bottom').is_null())
        )
        # .sort('ISP_vac_WR_wet' if vac else 'ISP_atm_WR_wet')
        .sort('ISP_vac' if vac else 'ISP_atm', descending=True)[0]
    )

def adapt_stack(diameter1, diameter2) -> list:
    i1 = diameters.index(diameter1)
    i2 = diameters.index(diameter2)
    # Make sure we're not going smaller for some reason (this would also break the adapter dict)
    assert i1 < i2
    if i2-i1 == 1:
        return [adapters[(diameter1, diameter2)]]
    elif i2 - i1 == 2:
        return [adapters[(diameter1, diameters[i1+1])], adapters[(diameters[i1+1], diameter2)]]
    elif i2 - i1 == 3:
        return [adapters[(diameter1, diameters[i1+1])], adapters[(diameters[i1+1], diameters[i2-1])], adapters[(diameters[i2-1], diameter2)]]
    else:
        raise Exception(f'Cant adapt from {diameter1} to {diameter2}')

def create_booster(parts:pl.DataFrame, payload_mass, upper_stage_diameter, *,
        lower_stage_diameter=None,
        upper_stage_deltaV=1000,
        bottom_stage_deltaV=3300,
        upper_stage_max_tanks=10,
        bottom_stage_max_tanks=15,
        upper_stage_min_TWR=.5,
        bottom_stage_min_TWR=1.5,
        upper_stage_resource=['LiquidFuel', 'Oxidizer'],
        bottom_stage_resource=['LiquidFuel', 'Oxidizer'],
        g=9.81,
    ) -> pl.DataFrame:
    if lower_stage_diameter is None:
        lower_stage_diameter == upper_stage_diameter
    if upper_stage_resource != ['LiquidFuel', 'Oxidizer'] or bottom_stage_resource != ['LiquidFuel', 'Oxidizer']:
        NotImplemented

    # wet mass, dry mass
    running_mass = np.array([payload_mass, payload_mass], dtype=float)
    parts = parts.with_columns(stage=pl.lit(0))

    def construct_stack(starting_diameter, diameter, vac, min_TWR):
        nonlocal running_mass

        # TODO: This only supports liquidFuel + Ox
        smallest_tank = parts.filter(c('provider').str.contains('Squad')).filter(
            c('diameter').list.contains(diameter) &
            (c('category') == 'FuelTank') &
            (c('maxLiquidFuel') > 0) &
            (c('maxOxidizer') > 0) &
            (~c('title').is_in(("R-12 'Doughnut' External Tank",))) &
            (~c('title').str.contains('(?i)adapter'))
        ).sort('wet_mass')[0]

        decoupler_diameter = starting_diameter
        decoupler_diameter2 = 'NOT A SIZE'
        if starting_diameter == 'size4':
            decoupler_diameter = 'size3'
        elif starting_diameter == 'size1p5':
            decoupler_diameter = 'size1'
            decoupler_diameter2 = 'size1p5'
        decoupler = parts.filter(
            (
                (c('diameter').list.contains(decoupler_diameter)) |
                (c('diameter').list.contains(decoupler_diameter2))
            ) &
            (c('category') == 'Coupling') &
            (c('title').str.contains('(?i)decouple')) &
            (c('provider') == 'Squad')
        )
        assert len(decoupler) == 1, 'found multiple or no decouplers for the given diameter'

        if starting_diameter == diameter:
            stack = [decoupler, None]
        else:
            try:
                stack = [
                    decoupler,
                    *[parts.filter(c('title') == i) for i in adapt_stack(starting_diameter, diameter)],
                    None
                ]
            except KeyError:
                raise Exception(f'Cant create booster: no adapter from {starting_diameter} to {diameter}')

        prev_engine_mass = 0
        # Construct the stage
        for _ in range((upper_stage_max_tanks if vac else bottom_stage_max_tanks)):
            running_mass -= prev_engine_mass
            running_mass += (smallest_tank['wet_mass'][0], smallest_tank['dry_mass'][0])
            # Takes into account it's own mass when calculating
            engine = optimal_engine(parts, *running_mass, diameter, vac, min_TWR)
            assert len(engine) == 1, 'Failed to create booster: No possible engine found'
            running_mass += engine['wet_mass'][0]
            prev_engine_mass = engine['wet_mass'][0]
            # Remove the engine from the stack, add another tank, and reoptimize the engine
            stack = [*stack[:-1], smallest_tank, engine]
            if (dv := deltaV(engine['ISP_vac' if vac else 'ISP_atm'][0], *running_mass, g)) >= (upper_stage_deltaV if vac else bottom_stage_deltaV):
                print(f'Calculated deltaV: {dv} for diameter {diameter}')
                stack = pl.concat(stack)
                if not vac:
                    stack = stack.with_columns(stage=pl.lit(1))
                # Make sure the decoupler is 1 stage offset
                stack[0, 'stage'] -= 1
                return stack

    # Upper stage
    for upper_diameter in diameters[diameters.index(upper_stage_diameter):]:
        stack = construct_stack(upper_stage_diameter, upper_diameter, True, upper_stage_min_TWR)
        if stack is None:
            continue
        break

    if stack is None:
        raise Exception('Failed to generate booster: Cannot create upper stage')

    # Bottom stage
    for lower_diameter in diameters[diameters.index(upper_diameter):]:
        bottom_stage  = construct_stack(upper_diameter, lower_diameter, False, bottom_stage_min_TWR)
        if bottom_stage is None:
            continue
        return pl.concat([stack, bottom_stage])

    raise Exception('Cant create booster: payload is too large')


def compile_header(
        save_name,
        description='',
        building='VAB',
        width=0, height=0, length=0,
    ) -> str:
    global arbitrary_id
    compiled = templates.header.format(
        save_name=save_name,
        description=description,
        building=building,
        width=width,
        height=height,
        length=length,
        save_persist_id=arbitrary_id,
    )
    arbitrary_id += 1
    return compiled

def read_existing_save(parts, file_path) -> tuple[list[float], str, str, int, int]:
    # Get the fairing details
    save = convert_list_to_dict(parse_cfg_file(file_path).as_list())
    matches = [m.value for m in parse('$.PART..part').find(save)]
    largest_fairing = parts.filter(c('name').is_in([m.split('_')[0].replace('.', '_') for m in matches if 'fairing' in m])).sort('dry_mass')[-1]
    if len(largest_fairing) == 0:
        raise Exception('Cant create booster: no fairing located in craft')
    largest_fairing_id = list(filter(lambda i: largest_fairing['name'][0] in i, matches))[0]
    pos = list(map(float, [m.value for m in parse(f'$.PART[?(@.part=={largest_fairing_id})].pos.`split(",", *, 2)`').find(save)][0]))
    # Go from the bottom of the fairing, not the top
    pos[1] += float([m.value for m in parse(f'$.PART[?(@.part=={largest_fairing_id})].attN[-1].`split(|,1,-1)`').find(save)][0])
    diameter = largest_fairing['diameter'][0][0]
    # Get the masses of all the parts and resources
    matches = [m.value for m in parse('$.PART..part').find(save)]
    mass = 0
    # Get the dry masses of all the parts
    for name in [m.split('_')[0].replace('.', '_') for m in matches]:
        part = parts.filter(c('name') == name)
        assert len(part) == 1, f'Error: Multiple parts found for the name {name}'
        mass += part['dry_mass'][0]
    # Get the masses of all the resources
    mass += sum(map(float, [m.value for m in parse('$.PART..RESOURCE[?(@.name==LiquidFuel)].amount').find(save)])) * .005
    mass += sum(map(float, [m.value for m in parse('$.PART..RESOURCE[?(@.name==Oxidizer)].amount').find(save)])) * .005
    mass += sum(map(float, [m.value for m in parse('$.PART..RESOURCE[?(@.name==Ore)].amount').find(save)])) * .01
    mass += sum(map(float, [m.value for m in parse('$.PART..RESOURCE[?(@.name==MonoPropellant)].amount').find(save)])) * .004
    # Get the masses of all the fairings
    # TODO: This slightly inaccurate
    mass += sum(map(float, [m.value for m in parse('$.PART..modMass').find(save)]))
    # last_decoupler_stage = max(map(int, re.finditer(r'dstg(?:\s+)?=(?:\s+)?(\d+)', existing_save)))
    last_decoupler_stage = max(map(int, [m.value for m in parse('$.PART..dstg').find(save)]))
    last_stage = max(map(int, [m.value for m in parse('$.PART..istg').find(save)]))
    return pos, diameter, largest_fairing_id, mass, last_decoupler_stage, last_stage

def compile_part(part:dict,
        x, y, z,
        node_top,
        node_bottom,
        stage,
        decoupler_stage,
        stage_index,
        *,
        linked_id:str|None=None,
        attached_top_id:str=NULL_PART,
        attached_bottom_id:str=NULL_PART,
        attPos0_0=0,
        attPos0_1=0,
        attPos0_2=0,
        sqor=None,
        sepI=None,
        # attach_mode,
        radially_attached=False,
    ) -> str:
    """ Returns the part id of this part, and then the compiled string """
    blocks = ''
    if part['maxLiquidFuel'] is not None:
        blocks += templates.liquid_fuel.format(amt=part['defaultLiquidFuel'], max=part['maxLiquidFuel'])
    if part['maxOxidizer'] is not None:
        blocks += templates.liquid_fuel.format(amt=part['defaultOxidizer'], max=part['maxOxidizer'])
    # TODO: all the other resources

    # Staging: https://forum.kerbalspaceprogram.com/topic/51800-any-doco-on-the-craft-file-format/?do=findComment&comment=1239388
    if sqor is None:
        sqor = stage if stage_index != -1 else -1
        if part['category'] == 'Tank':
            sqor = -1

    compiled = templates.part.format(
        part_name=part['name'].replace('_', '.'),
        part_id=part['id'],
        persist_id=part['id']+10_000,
        x=x, y=y, z=z,
        # Not quite sure what this is
        attPos01=0,
        linked=linked_id,
        attached_top=attached_top_id,
        attached_bottom=attached_bottom_id,
        node_top=str(node_top),
        node_bottom=str(node_bottom),
        link='' if linked_id is None else ('link = ' + linked_id),
        blocks=blocks,
        attPos0_0=attPos0_0,
        attPos0_1=attPos0_1,
        attPos0_2=attPos0_2,
        stage=stage,
        decoupler_stage=decoupler_stage,
        stage_index=stage_index,
        sqor=sqor,
        # I have no idea what this is or does, and I can't find any documentation for it.
        sepI=sepI or 1,
        attach_mode=int(radially_attached),
    )
    return compiled

def compile_parts(stack:pl.DataFrame, fairing_id, fairing_pos:tuple, last_decoupler_stage:int, last_stage:int):
    stack = (stack
        .with_row_index(name='id')
        .with_columns(id=c('id') + 1)
        .with_columns(part_id=c('name').str.replace('_', '.', literal=True, n=100) + '_' + c('id').cast(str))
    )

    y = fairing_pos[1]
    heights = []
    for i in stack.iter_rows(named=True):
        y -= i['node_stack_top'][1]
        heights.append(y)
        y -= abs(i['node_stack_bottom'][1] if i['variant_y_offsets'] is None else i['variant_y_offsets'][0])

    stack = stack.with_columns(
        prev_part_id=c('part_id').shift(1),
        next_part_id=c('part_id').shift(-1),
        height=np.linspace(fairing_pos[1]-1, 1, num=len(stack))
            if DEBUG else
            # np.array(list(it.accumulate(abs(stack['est_height'])/2, sub))) + fairing_pos[1] - stack['est_height'][0],
            np.array(heights),
    )
    display(stack)
    display(stack['name'].to_list())
    display(stack['title'].to_list())
    display(stack['height'].to_list())

    parts = []
    decoupler_stage = last_decoupler_stage
    for part in stack.iter_rows(named=True):
        if 'decoupler' in part['title'].lower() or 'stack separator' in part['title'].lower():
            decoupler_stage += 1
        parts.append(compile_part(
            part,
            fairing_pos[0], part['height'], fairing_pos[2],
            linked_id=part['next_part_id'],
            attached_top_id=(part['prev_part_id'] or fairing_id),
            node_top=part['node_stack_top'][1],
            node_bottom=part['node_stack_bottom'][1] if part['variant_y_offsets'] is None else part['variant_y_offsets'][1],
            stage=part['stage'] + last_stage + 1,
            decoupler_stage=decoupler_stage,
            # https://forum.kerbalspaceprogram.com/topic/51800-any-doco-on-the-craft-file-format/?do=findComment&comment=1239388
            stage_index=-1 if part['category'] == 'Tank' else 0,
        ))
        if 'decoupler' in part['title'].lower() or 'stack separator' in part['title'].lower():
            decoupler_stage += 1

    return ''.join(parts)

def generate_booster_save(parts, save_file, verbose=True, **kwargs):
    save_file = Path(save_file)
    if verbose: print(f'Generating Booster for payload named "{save_file.name[:-6]}"')
    fairing_pos, diameter, fairing_id, payload_mass, last_decoupler_stage, last_stage = read_existing_save(parts, save_file)
    if verbose: print(f'Payload has a mass of {payload_mass:.3f}T, and a {diameter} fairing at {fairing_pos}')
    try:
        stack = create_booster(parts, payload_mass, diameter, **kwargs)
    except Exception as err:
        # print(f'Failed to create booster: {err}')
        raise Exception(f'Failed to create booster') from err
    # stack = parts.filter(c('title').is_in(('Kerbodyne S3-7200 Tank', 'TD-06 Decoupler')))
    # stack = stack
    # stack = parts.filter(c('title') == 'Kerbodyne S3-7200 Tank')
    serialized = compile_parts(stack, fairing_id, fairing_pos, last_decoupler_stage, last_stage)

    with open(save_file) as f:
        existing_save = f.read()

    # This id mirrors the code in compile_parts()
    next_id = stack[0]["name"][0].replace("_", ".") + '_1'
    existing_save = existing_save.replace(f'part = {fairing_id}', f'part = {fairing_id}\n\tlink = {next_id}', 1)
    existing_save = re.sub(fr'(?<=part\ =\ {fairing_id})(?P<n>(?:(?:.|\n))*?attN\ =\ bottom,)(?P<o>[^\|]+)(?=(?:(?:.+)??\|)+)', fr'\g<n>{next_id}_0', existing_save)
    # Add 2 to all the existing stages, to offset for the stages we're adding to the bottom
    # existing_save = re.sub(r'istg(?:\s+)?=(?:\s+)?(\d+)', lambda m: f'istg = {int(m.group(1)) + 2}', existing_save)

    with open(str(save_file.parent / save_file.name[:-6]) + '_booster.craft', 'w') as f:
        f.write(existing_save + '\n' + serialized)
    # return existing_save + '\n' + parts

# def manually_attach_parts(parts:pl.DataFrame, save_file, verbose=True, **kwargs):
    # save_file = Path(save_file)
    # if verbose: print(f'Attaching part to payload named "{save_file.name[:-6]}"')
    # fairing_pos, diameter, fairing_id, payload_mass = read_existing_save(parts, save_file)
    # if verbose: print(f'Payload has a mass of {payload_mass:.3f}T, and a {diameter} fairing')
    # parts = compile_parts(parts, fairing_id, fairing_pos)

    # with open(save_file) as f:
    #     existing_save = f.read()
    # with open(str(save_file.parent / save_file.name[:-6]) + '_booster.craft', 'w') as f:
    #     f.write(existing_save + '\n' + parts)
    # return existing_save + '\n' + parts
