from inventory_plan import InventoryPlan
import numpy as np

inventory_plan = InventoryPlan()

v0 = np.concatenate((np.repeat(0, 3), np.repeat(1, 5), np.repeat(2, 4), np.repeat(3, 3), np.repeat(4, 2)))
v1 = np.concatenate((np.repeat(0, 10), np.repeat(1, 5), np.repeat(2, 2), np.repeat(3, 1), np.repeat(4, 1)))
v2 = np.concatenate((np.repeat(0, 13), np.repeat(1, 5), np.repeat(2, 1)))

seeds=[20, 21, 22, 23, 24, 25, 26, 27, 28]

inventory_plan.create_random_candidates(num_candidates=50000,
                                        seeds={'balls': 20,
                                               'bikes': 21,
                                               'blocks': 22,
                                               'books': 23,
                                               'coals': 24,
                                               'dolls': 25,
                                               'gloves': 26,
                                               'horses': 27,
                                               'trains': 28},
                                        gift_type_quantity_choices={'balls': v0,
                                                                    'bikes': v2,
                                                                    'blocks': v1,
                                                                    'books': v0,
                                                                    'coals': v2,
                                                                    'dolls': v1,
                                                                    'gloves': v0,
                                                                    'horses': v0,
                                                                    'trains': v1})

inventory_plan.calculate_expected_candidate_scores(trials=1000)

print inventory_plan.candidate_bags
