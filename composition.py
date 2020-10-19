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
    def __init__(self,f):
        self.f = f
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
    
def composable(f):
    return Composable(f)