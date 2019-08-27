import pystan
import pickle
from hashlib import md5

def StanModel_cache(model_name, extension=".stan", basepath="stan/", **kwargs):
    """Use just as you would `stan`"""
    fullname = basepath + model_name + extension
    with open(fullname, "r") as model_ondisk:
        model_code = model_ondisk.read()
    code_hash = md5(model_code.encode('ascii')).hexdigest()
    cache_fn = '{}cached-{}-{}.pkl'.format(basepath, model_name, code_hash)
    try:
        sm = pickle.load(open(cache_fn, 'rb'))
    except:
        sm = pystan.StanModel(model_code=model_code)
        with open(cache_fn, 'wb') as f:
            pickle.dump(sm, f)
    else:
        print("Using cached StanModel")
    return sm
