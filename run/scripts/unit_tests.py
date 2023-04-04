import numpy as np
import networkx as nx
from matplotlib import pyplot as plt
import unittest
import numpy as np
import networkx as nx
# from statsmodels.tsa.stattools import grangercausalitytests
# from lingam.var_lingam import VARLiNGAM
# from scipy.stats import pearsonr
# from scipy.stats import entropy
# from sklearn.metrics import mutual_info_score
from mos_auc_test import corr_test, mi_test, lingam_test, pairwise_granger_linear_test


def generate_causal_data(dag, sample_size, time_lag=1, noise_scale=0.1):
    """
    Generates causal data based on the input causal graph.

    Parameters:
    dag (nx.DiGraph): Directed acyclic graph representing the causal structure.
    sample_size (int): Number of samples to generate.
    noise_scale (float): Scale of the Gaussian noise added to the relationships.

    Returns:
    dict: A dictionary where the keys are node names, and the values are numpy arrays containing the generated data.
    """
    # Initialize data storage
    data = {node: None for node in dag.nodes}

    # Sort nodes topologically
    sorted_nodes = list(nx.topological_sort(dag))

    for node in sorted_nodes:
        parents = list(dag.predecessors(node))
        if not parents:
            # If node has no parents, generate random data
            data[node] = np.random.uniform(size=sample_size)
        else:
            # If node has parents, generate data based on parents and a linear relationship with noise
            weights = np.random.randn(len(parents))
            # non-gaussian noise
            noise = noise_scale * np.random.uniform(size=sample_size)
            data[node] = np.zeros(sample_size)
            data[node][time_lag:] = np.dot(np.column_stack([data[parent][:-time_lag] for parent in parents]),
                                           weights) + noise[:-time_lag]

    return data


# Include your generate_causal_data function here

class TestCausalDiscoveryMethods(unittest.TestCase):

    def setUp(self):
        # Define causal graph
        self.dag = nx.DiGraph()
        self.dag.add_edges_from([("A", "B")])

        # Generate causal data
        self.sample_size = 1000
        self.data = generate_causal_data(self.dag, self.sample_size)

    def test_granger_causality(self):
        test_data = np.stack([self.data['A'], self.data['B']], axis=1)
        score = pairwise_granger_linear_test(test_data)
        print("Granger score is ", score)
        assert abs(score) > 0, "Granger causality test failed for A -> B"

    def test_var_lingam(self):
        test_data = np.stack([self.data['A'], self.data['B']], axis=1)
        score = lingam_test(test_data)
        print("VAR score is ", score)
        assert score != 0, "VAR-LiNGAM test failed for A -> B"

    def test_correlation(self):
        test_data = np.stack([self.data['A'], self.data['B']], axis=1)
        corr = corr_test(test_data)
        print("Correlation is ", corr)
        assert abs(corr) > 0.5, "Correlation test failed for A -> B"

    def test_mutual_information(self):
        test_data = np.stack([self.data['A'], self.data['B']], axis=1)
        mi = mi_test(test_data)
        print("Mutual information is ", mi)
        assert mi >0.1, "Mutual information test failed for A -> B"


if __name__ == '__main__':
    unittest.main()


