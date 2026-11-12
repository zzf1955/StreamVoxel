"""Explicit, resumable learning-rate schedules for ControlNet continuations."""
import math


def learning_rate(step, *, peak_lr, warmup_steps=200, schedule='constant',
                  cosine_start_step=0, cosine_steps=0, min_lr=0.):
    if schedule=='constant':
        return peak_lr*min(1.,step/max(warmup_steps,1))
    if schedule!='cosine' or cosine_steps<=0 or cosine_start_step<0 or not 0<=min_lr<=peak_lr:
        raise ValueError('Invalid cosine learning-rate schedule')
    fraction=min(1.,max(0.,(step-cosine_start_step)/cosine_steps))
    return min_lr+(peak_lr-min_lr)*.5*(1+math.cos(math.pi*fraction))


def validate_resume_schedule(previous, current, checkpoint_step, *, allow_change=False):
    defaults=dict(lr_schedule='constant',cosine_start_step=0,cosine_steps=0,min_lr=0.)
    old={k:previous.get(k,v) for k,v in defaults.items()}
    new={k:current.get(k,v) for k,v in defaults.items()}
    if old==new:return
    if not allow_change:raise ValueError('Resume learning-rate schedule differs')
    if new['lr_schedule']!='cosine' or new['cosine_start_step']!=checkpoint_step:
        raise ValueError('New cosine schedule must start at the loaded checkpoint step')
    if new['cosine_steps']<=0 or current['steps']!=checkpoint_step+new['cosine_steps']:
        raise ValueError('New cosine continuation must have the requested exact duration')
    if not 0<=new['min_lr']<=current['lr']:raise ValueError('Invalid minimum LR')
