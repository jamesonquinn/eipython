import pyro

go = True
# go = False


NUM_BETWEEN_PRINT = 100

def printstuff(i,loss):
    print(f'epoch {i} loss = {loss};'+
        f' logsdrchat = {pyro.get_param_store()["logsdrchat"]};')
    print(f' erchat = {pyro.get_param_store()["erchat"]}')
    try:
        print(f' corrhat = {pyro.get_param_store()["corrhat"]}')
    except:
        pass
    #print(f' pnsml = {pyro.get_param_store()["precinct_newton_step_multiplier_logit"]}')

def demoprintstuff(i,loss):
    if (i % NUM_BETWEEN_PRINT) == 0:
        print(f'epoch {i} loss = {loss};')

        for item in ("mode_hat","gscale_hat","nscale_hat","narrower"):
            try:
                print(item,pyro.get_param_store()[item])
            except:
                pass


def printstuff2():
    print(f"ps2")


def getLaplaceParams():
    store = pyro.get_param_store()
    result = []
    for item in ("mode_hat","nscale_hat","gscale_hat","narrower",
                "echat","erchat","eprchat","corrhat"):
        try:
            result.append(store[item])
        except:
            pass

    return result
