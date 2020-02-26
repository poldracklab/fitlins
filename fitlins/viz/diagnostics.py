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

    def run(self, dm, checks=True, plots=False):
        results = {}
        if checks:
            for f in self._all_checks:
                results[f[1:]] = getattr(self, f)(dm)
        if plots:
            for f in self._all_plots:
                results[f[1:]] = getattr(self, f)(dm)

        return results
