import pystan
import pickle
from hashlib import md5

def StanModel_cache(model_name, extension=".stan", basepath="stan/", **kwargs):
    """Use just as you would `stan`"""
    fullname = basepath + model_name + extension
    with open(fullname, "r") as model_ondisk:
        model_code = model_ondisk.read()
    print(model_code)
    code_hash = md5(model_code.encode('ascii')).hexdigest()
    cache_fn = '{}cached-{}-{}.pkl'.format(basepath, model_name, code_hash)
    try:
        sm = pickle.load(open(cache_fn, 'rb'))
        print("Using cached StanModel")
        print(cache_fn)
    except:
        sm = None
    if sm is  None:
        print("compiling")
        sm = pystan.StanModel(file=fullname)
        print("compiled", cache_fn)
        with open(cache_fn, 'wb') as f:
            pickle.dump(sm, f)
    return sm
