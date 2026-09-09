"""Initialize one package; sync supporting files without editing student modules."""
import argparse
import json
from pathlib import Path
import shutil
from grade import grade
from policy import roster_map
ROOT=Path(__file__).resolve().parents[1]

def folder_for(login):
    mapping=roster_map(json.loads((ROOT/'roster.json').read_text())['students'])
    try:return ROOT/'students'/mapping[login.lower()]
    except KeyError:raise ValueError('Ask the teacher to register your GitHub login and name_studentnumber folder')

def copy_new(source,destination):
    if destination.is_symlink() or any(p.is_symlink() for p in destination.parents if p==ROOT or ROOT in p.parents):raise ValueError('Refusing symlink destination')
    if not destination.exists():
        destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,destination)

def sync(login,week,initialize=False):
    released=json.loads((ROOT/'course.json').read_text())['released_weeks']
    if week not in released:raise ValueError('Week has not been published')
    target=folder_for(login)
    if target.is_symlink():raise ValueError('Student folder cannot be a symlink')
    if initialize:
        if target.exists():raise ValueError('Package already exists; use sync (never reinitialize source)')
        for file in (ROOT/'template').rglob('*'):
            if file.is_file():copy_new(file,target/file.relative_to(ROOT/'template'))
    elif not (target/'src/mini_ml/__init__.py').is_file():raise ValueError('Initialize your package first')
    (target/'test_code').mkdir(exist_ok=True)
    for w in range(1,week+1):
        lesson=ROOT/'weeks'/f'week-{w:02}'
        copy_new(lesson/'demo.py',target/'test_code'/f'week_{w:02}.py')
        for file in (lesson/'data').glob('*.csv'):copy_new(file,target/'data'/f'week-{w:02}'/file.name)
    print(f'Ready: {target}\nRead weeks/week-{week:02}/lesson.md; manually integrate fragments. Existing source is unchanged.')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['init','sync','test']);p.add_argument('login');p.add_argument('week',type=int,nargs='?',default=1);a=p.parse_args()
    if a.action in ('init','sync'):sync(a.login,a.week,initialize=a.action=='init')
    else:
        if a.week not in json.loads((ROOT/'course.json').read_text())['released_weeks']:p.error('Week not published')
        raise SystemExit(grade(folder_for(a.login),ROOT/'tests',a.week))
