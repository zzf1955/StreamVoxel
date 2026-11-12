import math


def actual_time(mask_fraction):
    return 2 * math.asin(min(1., max(0., mask_fraction))) / math.pi


def reveal_fraction(u,schedule):
    if schedule=='linear':return u
    if schedule=='halfcos':return (1-math.cos(math.pi*u))/2
    if schedule=='reversequarter':return math.sin(math.pi*u/2)
    raise ValueError(schedule)


def quota_counts(total,steps,schedule):
    cumulative=[0]+[min(total,max(0,math.floor(total*reveal_fraction(s/steps,schedule)+1e-9))) for s in range(1,steps+1)]
    cumulative[-1]=total
    return [b-a for a,b in zip(cumulative,cumulative[1:])]


def curve(u,name):
    if name=='linear':return u
    if name=='quartercos':return 1-math.cos(math.pi*u/2)
    if name=='reversequarter':return math.sin(math.pi*u/2)
    if name=='halfcos':return (1-math.cos(math.pi*u))/2
    if name=='arccos':return 2/math.pi*math.acos(1-u)  # front-loaded: inverse of quartercos
    if name.startswith('front'):return 1-(1-u)**float(name[5:])  # front-loaded power: reveal most variables in the first steps
    if name.startswith('power'):return u**float(name[5:])
    raise ValueError(name)


def counts(total,spec):
    if spec['schedule']=='native_quarter':return None
    if spec.get('rounding')=='legacy':return quota_counts(total,spec['steps'],spec['schedule'])
    steps=spec['steps'];assert 1<=steps<=total
    cumulative=[0]+[s+math.floor((total-steps)*curve(s/steps,spec['schedule'])+1e-10) for s in range(1,steps+1)];cumulative[-1]=total
    q=[b-a for a,b in zip(cumulative,cumulative[1:])];assert min(q)>=1 and sum(q)==total
    return q
