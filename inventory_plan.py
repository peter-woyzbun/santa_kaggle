import numpy as np
import pandas as pd
from collections import Counter


class InventoryPlan(object):

    random_weights = {
        "horse": lambda: max(0, np.random.normal(5, 2, 1)[0]),
        "ball": lambda: max(0, 1 + np.random.normal(1, 0.3, 1)[0]),
        "bike": lambda: max(0, np.random.normal(20, 10, 1)[0]),
        "train": lambda: max(0, np.random.normal(10, 5, 1)[0]),
        "coal": lambda: 47 * np.random.beta(0.5, 0.5, 1)[0],
        "book": lambda: np.random.chisquare(2, 1)[0],
        "doll": lambda: np.random.gamma(5, 1, 1)[0],
        "blocks": lambda: np.random.triangular(5, 10, 20, 1)[0],
        "gloves": lambda: 3.0 + np.random.rand(1)[0] if np.random.rand(1) < 0.3 else np.random.rand(1)[0]
    }

    toy_types = {'ball': 'balls',
                 'bike': 'bikes',
                 'blocks': 'blocks',
                 'book': 'books',
                 'coal': 'coals',
                 'dolls': 'dolls',
                 'gloves': 'gloves',
                 'horse': 'horses',
                 'train': 'trains'}

    def __init__(self):

        self.available_inventory = Counter(book=1200,
                                           ball=1100,
                                           horse=1000,
                                           doll=1000,
                                           blocks=1000,
                                           train=1000,
                                           bike=500,
                                           gloves=200,
                                           coal=166)

        self.bags = list()
        self.candidate_bags = None
        self.expected_candidate_scores = None

    def create_random_candidates(self, num_candidates, seeds, gift_type_quantity_choices):
        """
        HA!


        :param num_candidates:
        :param seeds:
        :param gift_type_quantity_choices:
        :return:
        """
        candidate_bags = pd.DataFrame()
        for toy_type in self.toy_types.values():
            np.random.seed(seeds[toy_type])
            candidate_bags[toy_type] = np.random.choice(gift_type_quantity_choices[toy_type],
                                                        size=num_candidates,
                                                        replace=True)

        candidate_bags = candidate_bags.drop_duplicates()
        candidate_bags = candidate_bags.loc[
                         candidate_bags['balls'] + candidate_bags['bikes'] + candidate_bags['blocks'] +
                         candidate_bags['books'] + candidate_bags['coals'] + candidate_bags['dolls'] +
                         candidate_bags['gloves'] + candidate_bags['horses'] + candidate_bags['trains'] >= 3, :]\
            .reset_index(drop=True)

        self.candidate_bags = candidate_bags

    def add_candidate_bags(self, df):
        pass

    def expected_candidate_score(self, trials=1000, n_balls=0, n_bikes=0, n_blocks=0, n_books=0, n_coals=0, n_dolls=0,
                                 n_gloves=0, n_horses=0, n_trains=0):
        """
        To be !

        :param trials:
        :param n_balls:
        :param n_bikes:
        :param n_blocks:
        :param n_books:
        :param n_coals:
        :param n_dolls:
        :param n_gloves:
        :param n_horses:
        :param n_trains:
        :return:
        """

        balls = np.asarray([self.random_weights['ball']() for _ in range(trials * n_balls)])
        bikes = np.asarray([self.random_weights['bike']() for _ in range(trials * n_bikes)])
        blocks = np.asarray([self.random_weights['blocks']() for _ in range(trials * n_blocks)])
        books = np.asarray([self.random_weights['book']() for _ in range(trials * n_books)])
        coals = np.asarray([self.random_weights['coal']() for _ in range(trials * n_coals)])
        dolls = np.asarray([self.random_weights['doll']() for _ in range(trials * n_dolls)])
        gloves = np.asarray([self.random_weights['gloves']() for _ in range(trials * n_gloves)])
        horses = np.asarray([self.random_weights['horse']() for _ in range(trials * n_horses)])
        trains = np.asarray([self.random_weights['train']() for _ in range(trials * n_trains)])

        weight = np.concatenate((balls, bikes, blocks, books, coals, dolls, gloves, horses, trains))

        trial_id = np.concatenate((
            np.repeat(np.asarray(range(trials)) + 1, n_balls),
            np.repeat(np.asarray(range(trials)) + 1, n_bikes),
            np.repeat(np.asarray(range(trials)) + 1, n_blocks),
            np.repeat(np.asarray(range(trials)) + 1, n_books),
            np.repeat(np.asarray(range(trials)) + 1, n_coals),
            np.repeat(np.asarray(range(trials)) + 1, n_dolls),
            np.repeat(np.asarray(range(trials)) + 1, n_gloves),
            np.repeat(np.asarray(range(trials)) + 1, n_horses),
            np.repeat(np.asarray(range(trials)) + 1, n_trains)
        ))
        # Insert the results into a dataframe
        dt = pd.DataFrame()

        dt['trial_id'] = trial_id
        dt['weight'] = weight
        # Aggregate
        trials = dt.groupby(['trial_id']).agg(['sum']).reset_index()
        trials.columns = ['trial_id', 'weight']
        trials['score'] = trials['weight']
        trials.loc[trials.weight > 50, 'score'] = 0

        return trials['score'].mean(), trials['score'].var()

    def calculate_expected_candidate_scores(self, trials):

        for i in xrange(self.candidate_bags.shape[0]):
            expected_score, variance = self.expected_candidate_score(
                trials=trials,
                n_balls=self.candidate_bags.ix[i, 'balls'],
                n_bikes=self.candidate_bags.ix[i, 'bikes'],
                n_blocks=self.candidate_bags.ix[i, 'blocks'],
                n_books=self.candidate_bags.ix[i, 'books'],
                n_coals=self.candidate_bags.ix[i, 'coals'],
                n_dolls=self.candidate_bags.ix[i, 'dolls'],
                n_gloves=self.candidate_bags.ix[i, 'gloves'],
                n_horses=self.candidate_bags.ix[i, 'horses'],
                n_trains=self.candidate_bags.ix[i, 'trains']
            )
            self.candidate_bags.ix[i, 'expected_score'] = expected_score
            self.candidate_bags.ix[i, 'variance'] = variance

        self.candidate_bags = self.candidate_bags.sort_values(by='expected_score',
                                                              axis=0,
                                                              ascending=False).reset_index(drop=True)
        self.candidate_bags['candidate_id'] = self.candidate_bags.reset_index()['index']
        cols = ["candidate_id", "expected_score", "balls", "bikes", "blocks", "books", "coals", "dolls", "gloves", "horses",
                "trains"]
        self.candidate_bags = self.candidate_bags.ix[:, cols]

    def save_candidate_bags(self):
        pass
