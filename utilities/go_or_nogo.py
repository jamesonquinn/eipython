import pyro

go = True
# go = False

BREAK_NOW = False
# BREAK_NOW = True


NUM_BETWEEN_PRINT = 100

def printstuff(i,loss):
    print(f'epoch {i} loss = {loss};'+
        f' logsdrcstar = {pyro.get_param_store()["logsdrcstar"]};')
    print(f' ercstar = {pyro.get_param_store()["ercstar_raw"]}')
    try:
        print(f' corrstar = {pyro.get_param_store()["corrstar"]}')
    except:
        pass
    #print(f' pnsml = {pyro.get_param_store()["precinct_newton_step_multiplier_logit"]}')

def demoprintstuff(i,loss,mean_loss=None,*args):
    if (i % NUM_BETWEEN_PRINT) == 0:
        print(f'epoch {i} loss = {loss}; mean_loss= {mean_loss}; {args}')

        for item in ("mode_star","gscale_star","nscale_star","narrower","ltscale_star","ldfraw_star"):
            try:
                print(item,pyro.get_param_store()[item])
            except:
                pass


def printstuff2():
    print(f"ps2")


def getLaplaceParams():
    store = pyro.get_param_store()
    result = []
    for item in ("mode_star","nscale_star","gscale_star","narrower",
                "ecstar","ercstar","eprcstar","corrstar"):
        try:
            result.append(store[item])
        except:
            pass

    return result
