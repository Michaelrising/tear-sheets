import pickle
from yre.sim import Actual

actual = Actual()
actual =actual.read_pickle('../Paris/results/V15-l1-lgb-Y_hres_1-63D-126D-weight-None-search_pars-optuna-ic/Actuals/online Y_hres_1_bt.pkl')
actual.calc()
