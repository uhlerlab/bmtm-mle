import numpy as np
import os
from time import time
import pandas as pd
import itertools as itr
from tqdm import tqdm
from experiments.config import *


class ResultManager:
    def __init__(self, ps, num_trials, name):
        self.ps = ps
        self.num_trials = num_trials
        self.name = name

        # ensure that the directories for data and results are present
        os.makedirs(DATA_FOLDER, exist_ok=True)
        os.makedirs(RESULT_FOLDER, exist_ok=True)

        # create and save a dataframe of the appropriate shape to hold the results
        self.index_names = ["alg_name", "p", "trial_num"]
        self.column_names = ["result", "time"]
        if not os.path.exists(self.results_filename):
            self.save_empty_results()

    def save_empty_results(self):
        self.empty_result_df.to_pickle(self.results_filename)

    @property
    def empty_result_df(self):
        result_df = pd.DataFrame(columns=self.index_names + self.column_names)
        result_df.set_index(self.index_names, inplace=True)
        return result_df

    @property
    def data_filename(self):
        return os.path.join(DATA_FOLDER, f"{self.name}.npz")

    @property
    def results_filename(self):
        return os.path.join(RESULT_FOLDER, f"{self.name}.pkl")

    def get_data(self, overwrite=False):
        if os.path.exists(self.data_filename) and not overwrite:
            print(f"[ResultRunner.get_data] Loading data from {self.data_filename}")
            xs = np.load(self.data_filename)
            data = dict(xs.items())
        else:
            np.random.seed(12131)
            print(f"[ResultRunner.get_data] Generating data...")
            data = {str(p): np.random.normal(size=(self.num_trials, p)) for p in self.ps}
            print(f"[ResultRunner.get_data] Saving data to {self.data_filename}")
            np.savez(self.data_filename, **data)

        return data

    def _extract_trial_from_data(self, data, p, trial_num):
        return data[str(p)][trial_num, :]

    def get_results(self, algs, overwrite=False):
        # load the data and the results which have been processed so far
        data = self.get_data()
        if overwrite:
            old_result_df = self.empty_result_df
        else:
            old_result_df = pd.read_pickle(self.results_filename)

        # figure out which results still need to be obtained
        saved_result_keys = {tuple(index_val) for index_val in old_result_df.index}
        needed_results = set(itr.product(algs.keys(), self.ps, range(self.num_trials)))
        unsaved_result_keys = list(needed_results - saved_result_keys)

        # obtain the new results
        new_results = np.zeros([len(unsaved_result_keys), 2])
        for ix, (alg_name, p, trial_num) in enumerate(tqdm(unsaved_result_keys)):
            x = self._extract_trial_from_data(data, p, trial_num)
            start = time()
            m = algs[alg_name](x)
            elapsed = time() - start

            new_results[ix, 0] = m
            new_results[ix, 1] = elapsed

        # merge the new results with the old ones
        new_result_df = pd.DataFrame(
            new_results,
            index=pd.MultiIndex.from_tuples(unsaved_result_keys, names=self.index_names),
            columns=self.column_names
        )
        combined_result_df = pd.concat([old_result_df, new_result_df])
        combined_result_df.sort_index(inplace=True)

        # save the dataframe before returning
        combined_result_df.to_pickle(self.results_filename)
        return combined_result_df


if __name__ == '__main__':
    from algorithms.mean import mean, median
    rm = ResultManager(ps=[1000, 2000, 3000], num_trials=1000, name="rm_test")
    data = rm.get_data(overwrite=False)
    rm.save_empty_results()

    algs = {
        "mean": mean
    }
    res_df = rm.get_results(algs)
    print(res_df.shape)

    # after appending a new algorithm, we should only need to run the results for the new algorithm
    algs = {
        "mean": mean,
        "median": median
    }
    res_df = rm.get_results(algs)
    print(res_df.shape)

    # the overwrite keyword should be used to overwrite the old results, e.g. after modifying the algorithm.
    algs = {
        "mean": lambda x: np.mean(x) + 1,
        "median": lambda x: np.median(x) + 1
    }
    res_df = rm.get_results(algs, overwrite=True)
    print(res_df.shape)


