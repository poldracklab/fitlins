from np.lingalg import inv
from scipy import corrcoef


class Diagnostics:
    """ Warnings and plots generated from a first-level design matrix """

    def __init__(self):
        all_methods = [
            func for func in dir(self) if callable(getattr(self, func))]
        self._all_checks = [
            func for func in all_methods if func.startswith('_check')]
        self._all_plots = [
            func for func in all_methods if func.startswith('_plot')]

    @staticmethod
    def _check_covariance(dm):
        pass

    @staticmethod
    def _check_vif(df):
        cc = corrcoef(df.as_matrix(), rowvar=False)
        vif = inv(cc).diagonal()
        warn = df.columns[vif > 5].tolist()
        if warn:
            warn = "The following variables have a variance" \
                   "inflation factor > 5, indicating high multicolinearity: " \
                    f"{', '.join(df.columns[[0, 1]])}"
        else:
            warn = ""
        return {'result': vif, 'message': warn}

    def run(self, dm, checks=True, plots=False):
        results = {}
        if checks:
            for f in self._all_checks:
                results[f[1:]] = getattr(self, f)(dm)
        if plots:
            for f in self._all_plots:
                results[f[1:]] = getattr(self, f)(dm)

        return results
