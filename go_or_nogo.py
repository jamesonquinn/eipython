import pyro

go = True
# go = False


NUM_BETWEEN_PRINT = 10

def printstuff(i,loss):
    print(f'epoch {i} loss = {loss};'+
        f' logsdrchat = {pyro.get_param_store()["logsdrchat"]};')
    print(f' erchat = {pyro.get_param_store()["erchat"]}')
    print(f' pnsml = {pyro.get_param_store()["precinct_newton_step_multiplier_logit"]}')

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
