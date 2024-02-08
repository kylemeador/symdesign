"""Generate the code reference pages and navigation."""

from pathlib import Path

import mkdocs_gen_files

nav = mkdocs_gen_files.Nav()

src = Path(__file__).parent.parent
data_path = src / 'data'
print(data_path)
reference = Path('reference')

for path in sorted(src.rglob("*.py")):
    # Blacklisted files
    if 'third_party' in path.parts:
        continue
    elif 'extraction' in path.parts:
        continue
    elif 'interface_analysis' in path.parts:
        continue
    elif 'notebooks' in path.parts:
        continue
    elif 'conftest' in path.parts:
        continue
    elif data_path in path.parents:
        continue
    elif path.stem == 'run':
        continue

    module_path = path.relative_to(src).with_suffix('')
    # print(module_path, module_path.with_suffix(""))
    doc_path = module_path.with_suffix('.md')
    full_doc_path = reference / doc_path

    parts = tuple(module_path.parts)

    if parts[-1] == '__init__':
        parts = parts[:-1]
        if not parts:
            parts = (src.parts[-1],)
        doc_path = doc_path.with_name('index.md')
        full_doc_path = full_doc_path.with_name('index.md')
    elif parts[-1] == '__main__':
        continue

    nav[parts] = doc_path.as_posix()

    with mkdocs_gen_files.open(full_doc_path, 'w') as fd:
        ident = '.'.join(parts)
        fd.write(f'::: {ident}')

    mkdocs_gen_files.set_edit_path(full_doc_path, path)  # (name: str, edit-name: str)

with mkdocs_gen_files.open(reference / 'SUMMARY.md', 'w') as nav_file:
    nav_file.writelines(nav.build_literate_nav())
