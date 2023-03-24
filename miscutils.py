
# %cell name 'MISCUTILS'

import sys, os, json, glob, re, gc
import traceback, importlib, inspect, itertools, pandas, numpy
import pandas as pd
import numpy  as np

from pathlib         import Path
from os.path         import isdir, isfile
from traceback       import extract_tb
from threading       import get_ident
from collections.abc import Iterable 
from functools       import partial
from pprint          import pprint
from tempfile        import _get_candidate_names
from numpy           import percentile
from contextlib      import contextmanager
from time            import *
from datetime        import datetime, timezone, timedelta
from random          import random, choice, randint, shuffle, randrange
from matplotlib      import pyplot as plt, rcParams
from pandas          import DataFrame, Timestamp
undefined,null,false,true = None,None,False,True
Dict,List,Set,Tuple       = dict,list,set,tuple

# DEBUG #

loglevel = 1

def silent(*args): pass
def print0(*args): (print if loglevel >= 0 else silent)(f'<0> <{get_ident()}>', *args)
def print1(*args): (print if loglevel >= 1 else silent)(f'<1> <{get_ident()}>', *args)
def print2(*args): (print if loglevel >= 2 else silent)(f'<2> <{get_ident()}>', *args)
def print3(*args): (print if loglevel >= 3 else silent)(f'<3> <{get_ident()}>', *args)    

def most_recent_traceback(e=None):
    tb = e.__traceback__ if e else last(sys.exc_info())
    return extract_tb(tb,limit=-1)[0]

def most_recent_problem(e=None):
    tb = most_recent_traceback(e)
    return f'<{tb.name}> {tb.line}'

### FUNCTIONAL ###    

def  isnull(x): return not x
def notnull(x): return not not x
def  isnone(x): return x is None
def notnone(x): return x is not None

def   istext(x): return type(x) == str
def isnumber(x): return type(x) == int or type(x) == float

def flip(xy):
    x,y = xy
    return y,x

### PARSE ###

def extract_number(text, defval=0.):
    found = re.findall(r'(?=.*?\d)\d*[.,]?\d*', text) 
    found = found and found[-1]
    if found:
        found = found.replace(',','.')
        return float(found)
    return defval

# CONVERT #

def obj2dict(obj, keys=None, cond=lambda k: True):
    keys = keys or dir(obj)
    return dict([k,getattr(obj,k)] for k in keys if cond(k))

class dict2obj(object):
    def __init__(self, d):
        self._dict = d
        for k,v in d.items():
            if isinstance(v, (list, tuple)):
                setattr(self, k, [dict2obj(x) if isinstance(x, dict) else x for x in v])
            else:
                setattr(self, k, dict2obj(v) if isinstance(v, dict) else v)
    def __str__(self):
        return str(self._dict)   

# OOP #

def rename(newname):
    def decorator(f):
        f.__name__ = newname
        return f
    return decorator

class classproperty(object):
    def __init__(self, f):
        self.f = f
    def __get__(self, obj, owner):
        return self.f(owner)
    
def addmethod(cls):
    def decorator(func):
        setattr(cls, func.__name__, func)
        return func
    return decorator    

# CONTAINER #

def first(xx): 
    return next(iter(xx))

def second(xx): 
    i = iter(xx)
    next(i)
    return next(i)

def last(xx): 
    return first(reversed(xx))

def valat(i): 
    return (lambda xx: xx[i])

def valof(key): 
    return (lambda X: X.get(key))

def attrval(key): 
    return (lambda X: getattr(X, key))

def iterable(obj):
    return isinstance(obj, Iterable) and type(obj) != str

class ListContainer:
    def __init__(self, items):
        self.items = items
    def __getitem__(self, key):
        if callable(key): 
            return container([x for x in self.items if key(x)])
        if iterable(key): 
            return container([self.items[i] for i in key])        
        return self.items[key]
    def __contains__(self, item):
        if callable(item): 
            return next((True for x in self.items if item(x)), False)
        return item in self.items
    def __iter__(self):
        return iter(self.items)
    def __next__(self):
        return next(self.items)
    def __str__(self):
        return str(self.items)   
    
class DictContainer:
    def __init__(self, items):
        self.items = items
    def __getitem__(self, key):
        if callable(key): 
            return container(dict((k,v) for k,v in self.items.items() if key(k,v)))
        if iterable(key): 
            return container(dict((k,self.items[k]) for k in key))        
        return self.items[key]
    def __contains__(self, key):
        if callable(key): 
            return next((True for k,v in self.items.items() if key(k,v)), False)
        return key in self.items
    def __iter__(self):
        return iter(self.items.items())
    def __next__(self):
        return next(self.items.items())
    def __str__(self):
        return str(self.items)       

def container(items):
    if type(items) == dict: return DictContainer(items)
    if iterable(items):     return ListContainer(items)
    return DictContainer(obj2dict(items))

def chunks(arr, n):
    for i in range(0,len(arr),n):
        yield arr[i:i+n]
        
def flatten(*args):
    return itertools.chain(*args)
    
# THROTTLE #

class Throttle:    
    times = {}
    def __init__(self,fn,interval=None,timefn=None,depends=[]):
        if timefn is None: from time import time as timefn
        self.name     = fn.__name__
        self.fn       = fn
        self.t        = timefn()
        self.timefn   = timefn
        self.interval = interval or 30*60
        self.depends  = { name:Throttle.times.get(name,0) for name in depends }
        Throttle.times[self.name] = self.t
        print0(f"throttled '{self.name}' every {interval} seconds")
    def __call__(self,*args,**kwargs):
        if self.timefn() - self.t > self.interval:
            self.t      = self.timefn()
            should_skip = False
            for name in self.depends: 
                last_call_t        = Throttle.times.get(name,0)
                should_skip        = should_skip or last_call_t > self.depends[name] 
                self.depends[name] = last_call_t
            if not should_skip:
                Throttle.times[self.name] = self.t
                self.fn(*args,**kwargs)
    
def throttle(interval=None,timefn=None,depends=[]):
    def decorator(fn):
        return Throttle(fn,interval,timefn=timefn,depends=depends)
    return decorator

# COMPOSITION #

def spreadarg(fn):
    def proxy(*args):
        if len(args) >= 2: return fn(*args);
        if len(args) == 1: return fn(*args[0]);
        return fn();
    return proxy;

def compose(*ff):
    def composition(*args,**kwargs):
        for f in reversed(ff):
            x              = f(*args,**kwargs)
            is_args        = type(x) == tuple
            is_args_kwargs = is_args and len(x) == 2 and type(x[0]) == tuple and type(x[1]) == dict
            if is_args_kwargs:  args,kwargs = x
            elif is_args:       args,kwargs = x,{}
            else:               args,kwargs = (x,),{}
        return x
    return composition

class Composable:
    def __init__(self,x):
        if isinstance(x,Composable): self.f = x.f
        elif callable(x):            self.f = x
        else: raise Exception(f'invalid argument `{x}`')
    def __call__(self,*args,**kwargs):
        return self.f(*args,**kwargs)
    def __lshift__(self,other):              
        return Composable(compose(self,other))
    def __rshift__(self,other):              
        return Composable(compose(other,self))
    def __and__(self,other):
        def f(*args,**kwargs):
            return self(*args,**kwargs) and other(*args,**kwargs)
        return Composable(f)
    def __or__(self,other):
        def f(*args,**kwargs):
            return self(*args,**kwargs) or other(*args,**kwargs)
        return Composable(f)
    def __invert__(self):
        def f(*args,**kwargs):
            return not self(*args,**kwargs)
        return Composable(f)
    def __pow__(self, other):
        return Composable(partial(self.f,other))
    
def composable(f):
    return Composable(f)

# NETWORK #

def ip4addrs():
    from netifaces import interfaces, ifaddresses, AF_INET
    ip_list = []
    for interface in interfaces():
        ifaddrs = ifaddresses(interface)
        for link in ifaddrs.get(AF_INET,[]):
            ip_list.append(link['addr'])
    return ip_list

# MISC #

def time2str(t):
    return strftime("%b %d %Y %H:%M:%S", localtime(t))

def mtime(path):
    return os.path.getmtime(path)

def mtime2str(path):
    return time2str(mtime(path))

def getmethods(inst):
    return inspect.getmembers(inst, predicate=inspect.ismethod)
    
def getclasses(inst):
    return inspect.getmembers(inst, predicate=inspect.isclass)

def reloadmodule(module):
    if type(module) == str:
        module = sys.modules[module]
    importlib.reload(module)
    
def savetext(path, text, encoding='utf-8'):    
    with open(path,'w',encoding=encoding) as file:
        file.write(text)
    print0(path, mtime2str(path))
        
def loadtext(path,encoding='utf-8'):    
    print0(path, mtime2str(path))
    with open(path,'r',encoding=encoding) as file:
        return file.read()
    
def loadfolder(folder):
    items = []
    for file in Path(folder).rglob('*.json'):
        tmp    = eval(loadtext(file))
        items += tmp
    return items    
    
def validfilename(s):
    return ''.join([x if x.isalnum() else '_' for x in s])                 
    
def tempfilename():
    return next(_get_candidate_names());
    
def notifyme(msg, **kwargs):
    from requests import post
    url = 'https://api.pushover.net/1/messages.json?token=ajhpgiiie25dhehjek63q5w2p36r1r&user=umobvvsyqwdhxtgfmued18q6qxfee8';
    return post(url, data = { 'message': msg, **kwargs })
    
def showmetrics(prevmetrics, metrics):
    text = ''
    keys = [*set([*prevmetrics.keys(), *metrics.keys()])];
    keys.sort();
    for key in keys: 
        val  = metrics.get(key);
        pval = prevmetrics.get(key);
        if   not isnumber(pval) : ch = ''                
        elif not isnumber(val)  : ch = ''
        elif val  > pval        : ch = '↑'
        elif pval > val         : ch = '↓'
        else                    : ch = ''        
        text += f"{key}: {pval if val is None else val}{ch}\n"
    return text  

def splitfilepath(path):
    found = re.findall(r'(.+)-(\d+).(.+)',path)
    return found and found[0]

def getfilenum(path):
    path,num,sfx = splitfilepath(path)
    return int(num)

def nextfile(path):
    path,num,sfx = splitfilepath(path)
    pad,num      = len(num),int(num)
    return f'{path}-{(num+1):0>8d}.{sfx}'
    
def sortedfiles(path):
    paths = list(glob.glob(path)) 
    paths.sort(key=getfilenum)
    return paths   

def joinsources(target, sources, separator='\n'):    
    with open(target,'w') as fout:       
        for source in sources:
            print1(source, mtime2str(source))
            with open(source,'r') as fin:         
                if callable(separator): sep = separator(source=source)  
                else:                   sep = separator
                fout.write(sep)    
                fout.write(fin.read())
    print0(target, mtime2str(target))   

def pos2line(text,pos):
    lines = text.split('\n')
    for (i,line) in enumerate(lines):        
        if pos < len(line): 
            return i+1
        pos -= len(line)
        
def ctxsearch(path, regexpr, filter='*.*'):
    global loglevel
    ll,loglevel = loglevel,-1
    result = []
    for x in Path(path).rglob(filter):
        if isdir(x): continue
        text  = loadtext(x)
        found = re.search(regexpr,text)
        if found:
            pos  = found.span()[0]
            line = pos2line(text,pos)
            result.append((x,line))
    loglevel = ll
    return result
     
def getpopparam(ctx=None):  
    if not ctx: ctx = {};
    def popparam(param):
        if type(param) == list: 
            if ctx is not None:
                idx = ctx.get(id(param),0);
                ctx[id(param)] = idx+1;
                return param[idx];
            return param.pop(0);
        return param;
    return popparam; 

def getprogress():  
    time0 = time();
    def step(pos,length):
        progress = 100.0 * pos / length;
        elapsed  = time() - time0;
        left     = (length - (pos+1)) * elapsed / (pos+1);    
        predict  = time() + left + (4*60*60);
        return dict(
            progress = f'{round(progress,2)}%',           
            elapsed  = f'{round(elapsed)}s',
            left     = f'{round(left)}s',
            predict  = time2str(predict),   
        );
    return step; 

def paramlen(param):
    if type(param) == list: return len(param);
    return 1;

def param2str(param, sep=','):
    if type(param) == list: return sep.join(map(str,param));
    return str(param);
    
def print_list(list, name='list', leftlen=10, rightlen=10, suffix=''):
    print(name+':',len(list))
    print(*list[:leftlen],'...',*list[-rightlen:])
    if suffix is not None: print(suffix)
        
def pprint_list(list, name='list', leftlen=10, rightlen=10, suffix=''):
    print(name+':',len(list))
    pprint((*list[:leftlen],'...',*list[-rightlen:]))
    if suffix is not None: print(suffix)
        
def print_lens_stats(prefix, items, percentiles=[50,60,70,80,90,95,96,97,98,99]):
    lens = list(map(len,items));
    print(prefix+'-count:', len(lens));
    print(prefix+'-minlen:',min(lens));
    for pctl in percentiles:
        print(prefix+f'-pctl{pctl}:',percentile(lens,pctl));
    print(prefix+'-maxlen:',max(lens));    
    print();
    
# IPYNB #    

try:
    from IPython.core.interactiveshell import InteractiveShell
    InteractiveShell.ast_node_interactivity = 'all'
    'ast_node_interactivity:',InteractiveShell.ast_node_interactivity
    
    from IPython.core.magic import register_line_cell_magic
    from IPython.display    import Javascript, display, clear_output
    from pandas             import DataFrame
    from ipywidgets         import HTML, HBox, Button
    
    def ipynb2py(source, target, *keys, mode='w', **kwargs): 
        print0(source, mtime2str(source))
        from json import load
        with open(source) as notebook:
            data = load(notebook)
            with open(target,mode) as module:
                for key,value in kwargs.items():
                    module.write(f'{key} = {repr(value)};\n')
                module.write('\n')
                for cell in data['cells']:
                    if cell['cell_type'] == 'code' and cell['metadata'].get('name') in keys:
                        lines = cell['source']
                        if type(lines) == str: lines = lines.split('\n')
                        lines = filter(partial(re.match,'^(?!\s*[\!|\%])'),lines)    
                        code  = ''.join(lines)
                        module.write(code)
                        module.write('\n')
        print0(target, mtime2str(target))
        path = os.path.dirname(target)
        if path not in sys.path: 
            sys.path.append(path)

    def importipynb(notebook, *keys, module='ipynb', **kwargs):
        ipynb2py(notebook,module+'.py',*keys,**kwargs)
        if module in sys.modules:
            importlib.reload(sys.modules[module])
        importlib.import_module(module)  
        if module == 'ipynb': os.remove('ipynb.py')

    def getshownext(columns, functions, start=None, count=10, title=None):
        labels = None
        if type(columns) == dict: 
            labels    = [*columns.keys(), *functions.keys()]
            columns   = columns.values()
            functions = functions.values()
        maxlen = max(map(len,columns))
        if start is None: start = choice(range(maxlen))
        def shownext(start=start, count=count):
            clear_output(wait=True)
            if title is not None: display(HTML(value=f'<h4>{title}</h4>'))
            n = maxlen-start if start+count > maxlen else count
            sel = [(
                        i,
                        *[col[i] for col in columns],
                        *[fun(i) for fun in functions],
                    ) for i in range(start,start+n)]
            if labels: df = DataFrame(sel, columns=['index',*labels])
            else:      df = DataFrame(sel)
            display(df)
            button1 = Button(description=f'show prev {count}')
            button1.on_click(lambda _: shownext(start-count,count))
            button1.disabled = start <= 0
            button2 = Button(description=f'show next {count}')
            button2.on_click(lambda _: shownext(start+count,count))
            button2.disabled = start > maxlen-count
            display(HBox([button1,button2]))
        return shownext  

    get_running_cell_snippet = f'''
        function get_running_cell() {{
            var parent = element.parents('.cell');
            var index  = Jupyter.notebook.get_cell_elements().index(parent);
            var cell   = Jupyter.notebook.get_cell(index);
            return cell;
        }}
    '''     

    get_cell_by_name_snippet = f'''
        function get_cell_by_name(name) {{
            var cells = Jupyter.notebook.get_cells();
            var cell  = cells.find(({{ cell_type, metadata }}) => 
                            cell_type === 'code' && metadata.name === name);
            return cell;
        }}
    '''      
    
    execute_cell_sequence_snippet = f'''
        function execute_cell_sequence(names) {{
            var cell_indexes = [];
            var cells = Jupyter.notebook.get_cells();
            cells.forEach(({{ cell_type, metadata: {{name}} }},i) => {{
                if(cell_type === 'code' && names.includes(name))
                    cell_indexes.push(i);
            }});
            Jupyter.notebook.execute_cells(cell_indexes);
        }}
    '''      

    def update_notebook_metadata(**props):
        display(Javascript(f'''    
            var props = {json.dumps(props)};
            require([], () => Object.assign(Jupyter.notebook.metadata, props));     
        '''))    

    def update_cell_metadata(**props):
        display(Javascript(f'''    
           {get_running_cell_snippet}
            var props = {json.dumps(props)};
            require([], () => Object.assign(get_running_cell().metadata, props));        
        '''))      

    def execute_cell_sequence(names):
        display(Javascript(f'''    
            {execute_cell_sequence_snippet}
            var names = {json.dumps(names)};
            require([], () => execute_cell_sequence(names));        
        '''))           

    def initexeq(_, cell=None):
        display(Javascript(f'''  
            {get_running_cell_snippet}
            {execute_cell_sequence_snippet}
            require([], () => {{
                var initname = get_running_cell().metadata.name||'INIT';  
                var metadata = Jupyter.notebook.metadata;       
                var exec_seq = metadata.exec_sequence; 
                if(!exec_seq || exec_seq.length < 2) exec_seq = metadata.prev_exec_sequence;
                metadata.prev_exec_sequence = exec_seq;
                metadata.exec_sequence = [initname];
                if(!exec_seq || exec_seq.length < 2) return;
                var lastname = exec_seq[exec_seq.length-1];
                window.execute_last_sequence = () => execute_cell_sequence(exec_seq);
                var label = '%execute '+exec_seq.join(' ');
                var html = `<a href="#{{lastname}}" onclick="window.execute_last_sequence();">${{label}}</a>`;
                element.html(html);
            }});     
        '''))       

    def pushexeq(_, cell=None):
        display(Javascript(f'''  
            {get_running_cell_snippet}
            require([], () => {{
                var name = get_running_cell().metadata.name;  
                var {{exec_sequence}} = Jupyter.notebook.metadata;
                if(!exec_sequence || exec_sequence.includes(name)) return;
                exec_sequence.push(name);
            }});     
        '''))           
        
    def toggle(key, cell=None):
        display(Javascript(f'''  
            {get_running_cell_snippet}
            require([], () => {{
                var cell    = get_running_cell(); 
                var checked = cell.metadata['{key}'] !== false;            
                var id      = cell.metadata.name+'-{key}-switch';
                var html = `<div>
                    <input type="checkbox" id="${{id}}" name="${{id}}" ${{checked ? 'checked' : ''}}>
                    <label for="${{id}}">{key}</label>
                </div>`;
                element.html(html)
                var input = document.getElementById(id);
                input.onclick = () => {{
                    cell.metadata['{key}'] = input.checked;
                }};
            }});     
        '''))   
        
    def try_register_cell_magic():   
        def notebook(line, cell=None):
            key,value = re.split('\s+',line)
            props     = { key: eval(value) }
            update_notebook_metadata(**props)
        def cell(line, cell=None):
            key,value = re.split('\s+',line)
            props     = { key: eval(value) }
            update_cell_metadata(**props)
        def execute(line, cell=None):
            names = re.split('\s+',line)
            names = [*map(lambda name: name.strip('\'"'), names)]
            execute_cell_sequence(names)
        register_line_cell_magic(notebook)
        register_line_cell_magic(cell)
        register_line_cell_magic(execute)
        register_line_cell_magic(initexeq)
        register_line_cell_magic(pushexeq)
        register_line_cell_magic(toggle)

    try_register_cell_magic()   
    
except ImportError: pass
except Exception: pass

strptime = datetime.strptime
more_percentiles = [0.05,0.10,0.25,0.50,0.75,0.90,0.95,0.97,0.99]

ELAPSED = lambda t0,t=None: f'{timedelta(seconds=(t or time())-t0)}'
throttled_print  = throttle(5)(print)    
throttled_clear  = throttle(10*60)(clear_output)    
throttled_notify = throttle(30*60)(notifyme)    
def NOTIFY(*args, **kwargs):    
    throttled_print (*args, *kwargs.values())
    throttled_notify(*args, **kwargs)
    
def if_not_defined(scope, var, val):
    if var not in scope:
        scope[var] = val
    print(var+':',scope[var])    
    
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', None)
np.set_printoptions(threshold=sys.maxsize)
        
print('miscutils') 


