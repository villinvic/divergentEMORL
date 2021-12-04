from EMORL.Hub import Hub
from config.Loader import Default
from EMORL.plotting import plot_stats
import fire


class Runner(Default):
    def __init__(self, ip):
        self.ip = ip
        super(Runner, self).__init__()

    def __call__(self):
        for i in range(self.n_run):
            hub = Hub(self.ip, skip_init=True)
            hub.max_gen = self.max_gen
            hub()
            if self.do_plot:
                plot_stats(hub.population, 'results/experiment_'+str(i))
            del hub


if __name__ == '__main__':
    fire.Fire(Runner)