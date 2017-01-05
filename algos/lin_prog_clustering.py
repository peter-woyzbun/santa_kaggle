import pulp
import numpy as np
import pandas as pd

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

# prob = pulp.LpProblem('bag_arrangement', pulp.LpMaximize)

# ball_1 = pulp.LpVariable('ball_1', lowBound=0, cat='Integer', upBound=1)
# ball_2 = pulp.LpVariable('ball_2', lowBound=0, cat='Integer', upBound=1)


def random_gift_weights(num_each_type):
    weight_dict = dict()
    for key, func in random_weights.items():
        weight_dict[key] = list()
        for i in xrange(num_each_type):
            weight_dict[key].append(random_weights[key]())
        # weight_dict[key] = [random_weights[key]() * num_each_type]
    return weight_dict


def create_problem():
    prob = pulp.LpProblem('bag_arrangement', pulp.LpMaximize)
    gift_weights_dict = random_gift_weights(100)
    print gift_weights_dict
    prob_vars = list()
    for gift_type in gift_weights_dict.keys():
        gift_type_count = 0
        for gift_weight in gift_weights_dict[gift_type]:
            prob_vars.append((pulp.LpVariable('%s_%s' % (gift_type, gift_type_count), lowBound=0, cat='Integer', upBound=1), gift_weight))
            # prob += gift_weight * pulp.LpVariable('%s_%s' % (gift_type, gift_type_count), lowBound=0, cat='Integer', upBound=1)
            gift_type_count += 1
    var_expr = pulp.LpAffineExpression(prob_vars)
    prob += var_expr <= 50
    prob += var_expr
    prob.solve()
    for var in var_expr:
        print var.name
        print pulp.value(var)


class OptimalClustering(object):

    gift_types = {'ball': 'balls',
                 'bike': 'bikes',
                 'blocks': 'blocks',
                 'book': 'books',
                 'coal': 'coals',
                 'doll': 'dolls',
                 'gloves': 'gloves',
                 'horse': 'horses',
                 'train': 'trains'}

    def __init__(self, n_gifts_per_type, n_trials):
        self.n_gifts_per_type = n_gifts_per_type
        self.n_trials = n_trials
        self.candidate_bags = list()
        self.candidate_expected_weight_df = None
        self.weights_calculated = 0
        self.candidates_generated = 0

    @property
    def plural_gift_types(self):
        return self.gift_types.values()

    def random_gift_weights(self):
        weight_dict = dict()
        for key, func in random_weights.items():
            weight_dict[key] = list()
            for i in xrange(self.n_gifts_per_type):
                weight_dict[key].append(random_weights[key]())
        return weight_dict

    def create_candidate_bag(self):
        prob = pulp.LpProblem('bag_arrangement', pulp.LpMaximize)
        gift_weights_dict = self.random_gift_weights()
        prob_vars = list()
        for gift_type in gift_weights_dict.keys():
            gift_type_count = 0
            for gift_weight in gift_weights_dict[gift_type]:
                prob_vars.append((pulp.LpVariable('%s_%s' % (gift_type, gift_type_count), lowBound=0, cat='Integer',
                                                  upBound=1), gift_weight))
                gift_type_count += 1
        var_expr = pulp.LpAffineExpression(prob_vars)
        prob += var_expr <= 40 + np.random.randint(high=10, size=1, low=0)[0]
        prob += var_expr
        prob.solve()

        gift_type_counts = dict(books=0,
                                balls=0,
                                horses=0,
                                dolls=0,
                                blocks=0,
                                trains=0,
                                bikes=0,
                                gloves=0,
                                coals=0)
        for var in var_expr:
            if pulp.value(var) == 1.0:
                gift_type = var.name.split("_")[0]
                gift_type_counts[self.gift_types[gift_type]] += 1

        self.candidate_bags.append(gift_type_counts)

        self.candidates_generated += 1
        print "%s candidates generated." % self.candidates_generated

    def calculate_expected_candidate_values(self, n_trials):
        candidate_bags = self.candidate_bags_df()
        for i in xrange(self.candidate_bags_df().shape[0]):
            expected_score = self.expected_candidate_score(
                trials=n_trials,
                n_balls=candidate_bags.ix[i, 'balls'],
                n_bikes=candidate_bags.ix[i, 'bikes'],
                n_blocks=candidate_bags.ix[i, 'blocks'],
                n_books=candidate_bags.ix[i, 'books'],
                n_coals=candidate_bags.ix[i, 'coals'],
                n_dolls=candidate_bags.ix[i, 'dolls'],
                n_gloves=candidate_bags.ix[i, 'gloves'],
                n_horses=candidate_bags.ix[i, 'horses'],
                n_trains=candidate_bags.ix[i, 'trains']
            )
            candidate_bags.ix[i, 'expected_weight'] = expected_score

        self.candidate_expected_weight_df = candidate_bags

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

        balls = np.asarray([random_weights['ball']() for _ in range(trials * n_balls)])
        bikes = np.asarray([random_weights['bike']() for _ in range(trials * n_bikes)])
        blocks = np.asarray([random_weights['blocks']() for _ in range(trials * n_blocks)])
        books = np.asarray([random_weights['book']() for _ in range(trials * n_books)])
        coals = np.asarray([random_weights['coal']() for _ in range(trials * n_coals)])
        dolls = np.asarray([random_weights['doll']() for _ in range(trials * n_dolls)])
        gloves = np.asarray([random_weights['gloves']() for _ in range(trials * n_gloves)])
        horses = np.asarray([random_weights['horse']() for _ in range(trials * n_horses)])
        trains = np.asarray([random_weights['train']() for _ in range(trials * n_trains)])

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

        self.weights_calculated += 1
        print "%s expected weights calculated." % self.weights_calculated

        return trials['score'].mean()

    def run(self):
        for i in xrange(self.n_trials):
            self.create_candidate_bag()

    def candidate_bags_df(self):
        return pd.DataFrame(self.candidate_bags)


test_clustering = OptimalClustering(n_gifts_per_type=25, n_trials=4000)
test_clustering.run()

test_clustering.calculate_expected_candidate_values(n_trials=1000)

test_clustering.candidate_expected_weight_df.to_csv('candidate_bags_4.csv')

print test_clustering.candidate_expected_weight_df



