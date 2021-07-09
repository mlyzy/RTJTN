from __future__ import absolute_import

from models.bert_PRE import *

__factory = {
    'transfromer': Transfromer_Baseline,
    'transfromer_2class': Transfromer_Baseline_2class,
}

def get_names():
    return __factory.keys()

def init_model(name, *args, **kwargs):
    if name not in __factory.keys():
        raise KeyError("Unknown model: {}".format(name))
    return __factory[name](*args, **kwargs)
