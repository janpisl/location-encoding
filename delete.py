import re #python regular expression matching module

path = 'planted_trees_db.py'
with open(path, 'r') as f_orig:
    script = re.sub(r'# In\[.*\]:\n','', f_orig.read())
with open(path,'w') as fh:
    fh.write(script)