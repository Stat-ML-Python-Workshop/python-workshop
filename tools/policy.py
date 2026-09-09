"""Map authenticated PR authors to unique course folders."""
import re
TEACHER='tttt1314'

def roster_map(roster):
    if not isinstance(roster,list):raise ValueError('students must be a list')
    mapping={};folders=set()
    for entry in roster:
        if not isinstance(entry,dict) or set(entry)!={'github_login','folder'}:raise ValueError('Each student needs github_login and folder')
        login,folder=entry['github_login'],entry['folder']
        if not isinstance(login,str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9-]{0,38}',login):raise ValueError('Invalid GitHub login')
        if not isinstance(folder,str) or len(folder)>100 or not re.fullmatch(r'[\w-]+_[0-9]+',folder):raise ValueError('Folder must be name_studentnumber')
        if login.lower() in mapping or folder.casefold() in folders:raise ValueError('Duplicate login or folder')
        mapping[login.lower()]=folder;folders.add(folder.casefold())
    return mapping

def validate(author,branch,head_repository,base_repository,files,roster,released):
    mapping=roster_map(roster)
    if head_repository!=base_repository:raise ValueError('Use a branch in the class repository')
    if author.lower()==TEACHER:
        if not branch.startswith('instructor/'):raise ValueError('Teacher PRs use instructor/ branches')
        return {'kind':'teacher','student':'','folder':'','week':max(released,default=0)}
    if author.lower() not in mapping:raise ValueError('Teacher must register your GitHub login first')
    match=re.fullmatch(re.escape(author)+r'/week-(\d{2})',branch,re.IGNORECASE)
    if not match:raise ValueError('Branch must be <GitHub login>/week-XX')
    week=int(match[1])
    if week not in released:raise ValueError('Week has not been released')
    folder=mapping[author.lower()];prefix=f'students/{folder}/'
    if not files or any(not f.startswith(prefix) or '..' in f.split('/') for f in files):raise ValueError('Change only your registered student folder')
    if any('test_code' in f.split('/') for f in files):raise ValueError('test_code is local-only; unstage these files')
    return {'kind':'student','student':author,'folder':folder,'week':week}
