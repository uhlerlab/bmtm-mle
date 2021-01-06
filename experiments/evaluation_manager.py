import matplotlib.pyplot as plt
import seaborn as sns
import os
from experiments.result_manager import ResultManager
from experiments.config import *
sns.set()

# A dictionary mapping from each algorithm to labels allows the labels to change in the future.
ALGS2LABELS = {
    "mean": "Mean",
    "median": "Median"
}

# A dictionary mapping from each algorithm to a color allows the colors to be kept consistent between plots.
ALGS2COLORS = dict(zip(ALGS2LABELS.keys(), sns.color_palette()))


def compute_loss(res, true_val):
    return (res - true_val)**2


class EvaluationManager:
    def __init__(self, result_manager: ResultManager):
        self.result_manager = result_manager
        self.name = self.result_manager.name
        os.makedirs(self.figure_folder, exist_ok=True)

    @property
    def figure_folder(self):
        return os.path.join(FIGURE_FOLDER, self.name)

    def plot_p_vs_accuracy(self, algs):
        plt.clf()
        res_df = self.result_manager.get_results(algs)
        true_val = 0
        losses = res_df.groupby(["alg_name"]).apply(lambda res: compute_loss(res, true_val))
        loss_means = losses.groupby(["alg_name", "p"])["result"].mean()
        loss_stds = losses.groupby(["alg_name", "p"])["result"].std()
        ps = self.result_manager.ps

        for alg_name in algs.keys():
            alg_means = loss_means[loss_means.index.get_level_values("alg_name") == alg_name]
            alg_stds = loss_stds[loss_stds.index.get_level_values("alg_name") == alg_name]
            plt.plot(
                ps,
                alg_means,
                label=ALGS2LABELS[alg_name],
                color=ALGS2COLORS[alg_name]
            )
            plt.fill_between(
                ps,
                alg_means - alg_stds,
                alg_means + alg_stds,
                color=ALGS2COLORS[alg_name],
                alpha=0.2
            )

        plt.xlabel("p")
        plt.ylabel("Squared error")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.figure_folder, "accuracy.png"))

    def plot_p_vs_time(self, algs):
        plt.clf()
        res_df = self.result_manager.get_results(algs)
        means = res_df.groupby(["alg_name", "p"])["time"].mean()
        stds = res_df.groupby(["alg_name", "p"])["time"].std()
        ps = self.result_manager.ps

        for alg_name in algs.keys():
            alg_means = means[means.index.get_level_values("alg_name") == alg_name]
            alg_stds = stds[stds.index.get_level_values("alg_name") == alg_name]
            plt.plot(
                ps,
                alg_means,
                label=ALGS2LABELS[alg_name],
                color=ALGS2COLORS[alg_name]
            )
            plt.fill_between(
                ps,
                alg_means-alg_stds,
                alg_means+alg_stds,
                color=ALGS2COLORS[alg_name],
                alpha=0.2
            )
        plt.xlabel("p")
        plt.ylabel("Computation Time")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.figure_folder, "time.png"))


if __name__ == '__main__':
    from algorithms.mean import mean, median
    from experiments.result_manager import ResultManager

    rm = ResultManager(ps=[1000, 2000, 3000, 4000, 5000, 6000], num_trials=1000, name="em_test")
    algs = {
        "mean": mean,
        "median": median
    }
    res_df = rm.get_results(algs)

    em = EvaluationManager(rm)
    em.plot_p_vs_accuracy(algs)
    em.plot_p_vs_time(algs)

