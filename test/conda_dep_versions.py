import io
import json
import os.path
import string
import subprocess
import yaml

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from tqdm import tqdm


def digit_keeper() -> defaultdict:
    return defaultdict(type(None), dict(zip(map(ord, string.digits), string.digits)))  # '0123456789'


def digit_remover() -> defaultdict:
    non_numeric_chars = string.printable[10:]
    # 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ \t\n\r\x0b\x0c'
    keep_chars = dict(zip(map(ord, non_numeric_chars), non_numeric_chars))

    return defaultdict(type(None), keep_chars)


keep_digit_table = digit_keeper()
remove_digit_table = digit_remover()


def version_as_tuple(v):
    version = []
    for id_ in v.split('.'):
        try:
            version.append(int(id_))
        except ValueError:
            version.append(int(id_.translate(keep_digit_table)))
    return tuple(version)
    # return tuple(map(int, v.split('.')))


def shell(cmd):
    proc = subprocess.run(cmd, shell=True, capture_output=True)
    return proc.stdout.decode('utf-8')


def get_history(package):
    if channels:
        cmd = f'conda search -c {" -c ".join(channels)} -q {package} --info --json'
    else:
        cmd = f'conda search -q {package} --info --json'
    # input(cmd)
    txt = shell(cmd)
    h = defaultdict(set)
    try:
        d = json.loads(txt)
    except json.decoder.JSONDecodeError:
        print(f"The package {package} didn't work during search command: {cmd}")
    else:
        try:
            for vv in d.values():
                for x in vv:
                    h[version_as_tuple(x['version'])].add(
                        datetime.fromtimestamp(x.get('timestamp', 0) / 1e3)
                    )
            h = {vers: min(dates) for vers, dates in h.items()}
        except TypeError as error:
            print(f'Found error {error} in results dictionary:{d}')
        else:
            return package, h
    return package, h


# metayaml = """
#     - boto3
#     - pandas >=0.25
#     - python >=3.8
# """
here = os.path.dirname(__file__)
os.chdir(here)
project_yml = '../conda_env.yml'
metayaml = io.open(project_yml)
reqs = yaml.safe_load(metayaml)
# input(reqs)
interested_headers = ['dependencies']
# all_pkgs = {item for header, items in reqs.items() if header in interested_headers for item in items}
channels = set()
all_pkgs = set()
for header, items in reqs.items():
    if header in interested_headers:
        for item in items:
            if isinstance(item, str):
                all_pkgs.add(item)
    elif header == 'channels':
        channels.update(items)

pip = []
for header, items in reqs.items():
    if header in interested_headers:
        for item in items:
            if isinstance(item, dict):
                for key, value in item.items():
                    if key == 'pip':
                        pip.extend(value)

# input(all_pkgs)
# input(pip)
all_pkgs = [pkg_version.split('=')[0].strip() for pkg_version in sorted(all_pkgs.union(pip))]
package_str = '\n  - '.join(all_pkgs)
cpus_to_use = os.cpu_count() // 2  # Leave half for the computer
# cpus_to_use = 2  # Leave half for the computer
num_packages = len(all_pkgs)
input(f'Performing check on {num_packages} packages:\n  - {package_str}\n'
      f'Using {cpus_to_use} cpus. Press Enter to continue\n')
with ThreadPoolExecutor(max_workers=cpus_to_use) as pool:
    history = dict(tqdm(pool.map(get_history, all_pkgs), total=num_packages))
# Without ThreadPoolExecutor()
# history = dict(tqdm(map(get_history, all_pkgs), total=num_packages))

# {v: f'{t:%Y-%m-%d}' for v, t in history['pandas'].items()}

# asof = datetime.now() - timedelta(weeks=2*52)
asof = datetime.now() - timedelta(weeks=4)
# asof = datetime.now() - timedelta()

# new = {
#     name: max([(vers, t) for vers, t in v.items() if t < asof])
#     for name, v in history.items()
new = {}
for name, v in history.items():
    versions_to_timestamp = [(vers, t) for vers, t in v.items() if t < asof]
    # print(versions_to_timestamp)
    new[name] = max(versions_to_timestamp)

print(f'# as of {asof:%Y-%m-%d}')
for name, (vers, t) in new.items():
    print(f'  - {name}=={".".join(map(str, vers))} # released on {t:%Y-%m-%d}')
