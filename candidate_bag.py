from collections import Counter
import numpy as np
import random


class CandidateBag(object):

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

    def __init__(self, n_samples=1000):
        self.n_samples = n_samples
        self.gift_type_counts = Counter(book=0,
                                        ball=0,
                                        horse=0,
                                        doll=0,
                                        blocks=0,
                                        train=0,
                                        bike=0,
                                        gloves=0,
                                        coal=0)
        self.last_gift_added = None
        self.final_expected_weight = 0

    def add_random_gift(self):
        """
        Add a random gift type to the bag's current inventory.

        """

        gift_choices = ['ball', 'horse', 'book', 'doll', 'blocks', 'gloves']

        if self.gift_type_counts['bike'] < 1:
            gift_choices.append('bike')
        if self.gift_type_counts['train'] < 3:
            gift_choices.append('train')
        if self.gift_type_counts['coal'] < 1:
            gift_choices.append('coal')

        random_gift_type = random.choice(gift_choices)
        self.gift_type_counts[random_gift_type] += 1
        self.last_gift_added = random_gift_type

    def remove_last_gift(self):
        """
        Remove the last gift that was added by decrementing the
        gift type counter accordingly.

        """
        if self.last_gift_added:
            self.gift_type_counts[self.last_gift_added] -= 1

    @property
    def expected_weight(self):
        gifts = list(self.gift_type_counts.elements())
        sample_weights = list()
        for i in xrange(self.n_samples):
            sample_weight = sum([self.random_weights[gift_type]() for gift_type in gifts])
            if sample_weight <= 50:
                sample_weights.append(sample_weight)
            else:
                sample_weights.append(0)
        return np.average(sample_weights)

    def populate_randomly(self):
        while True:
            old_expected_weight = self.expected_weight
            self.add_random_gift()
            new_expected_weight = self.expected_weight
            if new_expected_weight < old_expected_weight:
                self.remove_last_gift()
                self.final_expected_weight = old_expected_weight
                break


candidate_bag = CandidateBag(n_samples=1000)

candidate_bag.populate_randomly()

print candidate_bag.gift_type_counts
print candidate_bag.expected_weight