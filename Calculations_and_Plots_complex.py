import numpy as np
import matplotlib.pyplot as plt
import LBP_FactorGraphs_complex as lbp

'''
    Implementing the SumProduct and calculating marginals and free energies
    Also plotting and graph visualizing
'''


def calc_n_plot(g, t_max, epsilon, vis, single_node, single_node_from_factors, compare, free_energies, joint_flag):
    # Implementing the algorithm
    g.vis_graph(vis)
    #z = g.exact_partition()
    #F = - np.log(np.abs(z))
    beliefs, factor_beliefs, t_max = g.sum_product(t_max, epsilon)
    beliefs_from_factor_beliefs = []
    beliefs_from_factors_to_energy = np.zeros([g.node_count, t_max + 1, 2], dtype=complex)
    f_mean_field = np.ones(t_max + 1, dtype=float)
    f_mean_field_from_factor_beliefs = np.ones(t_max + 1, dtype=float)
    f_bethe = np.ones(t_max + 1, dtype=float)

    # Initialization of single node beliefs calculated from factor beliefs
    for i in range(g.node_count):
        beliefs_from_factor_beliefs.append({})
        for item in g.nodes[i][2]:
            beliefs_from_factor_beliefs[i][item] = np.ones([t_max + 1, 2], dtype=complex)

    # Calculating single node beliefs from factor beliefs
    for t in range(t_max + 1):
        for i in range(g.node_count):
            for item in g.nodes[i][2]:
                beliefs_from_factor_beliefs[i][item][t] *= (np.einsum(factor_beliefs[item][t], range(g.node_count), [i]) / np.sum(factor_beliefs[item][t]))
            beliefs_from_factors_to_energy[i, t] += beliefs_from_factor_beliefs[i][item][t]

    # Calculating the approximated joint distribution for a tree like graph
    joint = []
    joint_normalization = []
    KL_divergence = []
    shape_of_joint = []
    for i in range(g.node_count):
        shape_of_joint.append(g.nodes[i][1])
    real_joint = np.ones(shape_of_joint, dtype=complex)
    for item in factor_beliefs:
        real_joint *= g.factors[item][1]
    real_joint /= np.sum(real_joint)
    for t in range(t_max):
        joint.append(np.ones(shape_of_joint, dtype=complex))
        for item in factor_beliefs:
            joint[t] *= factor_beliefs[item][t]
        for i in range(g.node_count):
            joint[t] /= (lbp.Graph.broadcasting(g, np.array(beliefs[i, t]), np.array([i]))) ** (len(g.nodes[i][2]) - 1)
        #joint[t] /= np.sum(joint[t])                        # adding normalization constant to the approximated joint
        KL_divergence.append(np.abs(np.sum(joint[t] * np.log(joint[t] / real_joint))))
        joint_normalization.append(np.abs(np.sum(joint[t])))


    # Calculating free energies
    for t in range(t_max + 1):
        factor_beliefs_for_F = {}
        for item in factor_beliefs:
            factor_beliefs_for_F[item] = factor_beliefs[item][t]
        f_mean_field[t] = g.mean_field_approx_to_F(beliefs[:, t])
        f_mean_field_from_factor_beliefs[t] = g.mean_field_approx_to_F(beliefs_from_factors_to_energy[:, t][:])
        f_bethe[t] = g.bethe_approx_to_F(beliefs[:, t], factor_beliefs_for_F)

    # Plotting Data
    if single_node:
        plt.figure()
        plt.title('Single node marginals')
        for i in range(g.node_count):
            plt.plot(range(t_max + 1), np.abs(beliefs[i]), 'o')
        plt.show()

    if single_node_from_factors:
        plt.figure()
        plt.title('Single node marginals calculated from factor beliefs')
        for i in range(g.node_count):
            for item in g.nodes[i][2]:
                plt.plot(range(t_max + 1), np.abs(beliefs_from_factor_beliefs[i][item][:, 0]), 'o')
                plt.plot(range(t_max + 1), np.abs(beliefs_from_factor_beliefs[i][item][:, 1]), 'o')
        plt.show()

    if compare:
        j = 0
        object = 'F,0,1'
        plt.figure()
        plt.title('comparing node marginals of a')
        plt.plot(range(t_max + 1), np.abs(beliefs[j]), 's')
        plt.plot(range(t_max + 1), np.abs(beliefs_from_factor_beliefs[j][object][:, 0]), 'o')
        plt.plot(range(t_max + 1), np.abs(beliefs_from_factor_beliefs[j][object][:, 1]), 'o')
        #plt.plot(range(t_max), beliefs_from_factors_to_energy[j][0:t_max, 0], 's')
        #plt.plot(range(t_max), beliefs_from_factors_to_energy[j][0:t_max, 1], 's')
        #plt.legend(['a(1)', 'a(-1)', 'a_ha(1)', 'a_ha(-1)', '1', '-1'])
        #plt.show()

        delta0 = np.abs(beliefs[j][0:t_max + 1, 0] - beliefs_from_factor_beliefs[j][object][:, 0])
        delta1 = np.abs(beliefs[j][0:t_max + 1, 1] - beliefs_from_factor_beliefs[j][object][:, 1])


        plt.figure()
        plt.title('Error between marginal calculation over node 0')
        plt.plot(range(t_max + 1), delta0, 'o')
        plt.plot(range(t_max + 1), delta1, 'o')
        plt.show()

    if free_energies:
        plt.figure()
        plt.title('Free energies')
        plt.plot(range(t_max + 1), f_mean_field, 's')
        plt.plot(range(t_max + 1), f_mean_field_from_factor_beliefs, 's')
        plt.plot(range(t_max + 1), f_bethe, 'o')
        plt.plot(range(t_max + 1), np.ones(t_max + 1, dtype=float) * F)
        #plt.ylim((F - np.abs(F - f_bethe[t_max]), f_bethe[t_max] + np.abs(F - f_bethe[t_max])))
        plt.legend(['F_mf', 'F_mf_from_factor_beliefs', 'F_Bethe', 'F_exact'])
        plt.show()

    if joint_flag:
        plt.figure()
        plt.title('Approximated joint normailzation and KL divergence')
        plt.plot(range(t_max), joint_normalization, 'o')
        plt.plot(range(t_max), KL_divergence, 's')
        plt.plot(range(t_max), np.ones(t_max))
        plt.plot(range(t_max), np.zeros(t_max))
        plt.yscale('log')
        plt.legend(['Normalization', 'KL divergence'])
        plt.show()

    return t_max
