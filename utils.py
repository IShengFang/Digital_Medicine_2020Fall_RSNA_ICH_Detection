import os

def check_and_make_dir(path):
    print('checking... ', path)
    dirs = path.split('/')
    if path[:2]=='./':
        path = path[:2]
    elif path[0]=='/':
        path = '/'
    else:
        path = ''
    for _dir in dirs:
        path = os.path.join(path, _dir)
        print('check', path, os.path.isdir(path))
        if not os.path.isdir(path):
            print('mkdir', path)
            os.mkdir(path)