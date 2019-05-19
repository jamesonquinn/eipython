import pyro

go = True
# go = False


def printstuff(i,loss):
    print(f'epoch {i} loss = {loss};'+
        f' logsdrchat = {pyro.get_param_store()["logsdrchat"]};')
    print(f' erchat = {pyro.get_param_store()["erchat"]}')
    print(f' pnsml = {pyro.get_param_store()["precinct_newton_step_multiplier_logit"]}')

def demoprintstuff(i,loss):
    print(f'epoch {i} loss = {loss};')
    try:
        print(f' mode_hat = {pyro.get_param_store()["mode_hat"]};')
        print(f' nscale_hat = {pyro.get_param_store()["nscale_hat"]}')
        print(f' gscale_hat = {pyro.get_param_store()["gscale_hat"]}')
    except:
        pass


def printstuff2():
    print(f"ps2")
