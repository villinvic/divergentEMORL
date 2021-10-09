import numpy as np


def is_dominated(x_scores, y_scores, epsilon):
    assert len(x_scores) == len(y_scores)

    for i in range(len(x_scores)):
        if i != 0:
            eps = 0
        else:
            eps = epsilon

        if x_scores[i] > y_scores[i] + eps:
            return True
        elif x_scores[i] < y_scores[i]:
            return False

    return False

def argsort_with_order(seq):

    seqq = np.concatenate([seqq[:, np.newaxis] for seqq in seq[:,:,0]], axis=1)

    names_l = [str(i) for i in range(len(seq))] # +1
    f = ', '.join(['f8' for _ in range(len(seq))]) # +1
    names = ', '.join(names_l)

    with_fields = np.core.records.fromarrays(seqq.transpose(), names=names, formats=f)

    return list(np.argsort(with_fields, order=tuple(names_l)))




def ND_sort(scores, n_objectives=2, epsilon=0):
    """
    builds frontiers, descending sort
    """
    frontiers = [[]]
    assert n_objectives > 1
    indexes = np.array(list(reversed(argsort_with_order(scores))))
    for index in indexes:
        x = len(frontiers)
        k = 0
        while True:
            dominated = False
            for solution in frontiers[k]:
                tmp = True
                for objective_num in range(1, n_objectives):
                    if is_dominated(scores[objective_num][index], scores[objective_num][solution], epsilon):
                        tmp = False
                        break
                dominated = tmp
                if dominated:
                    break
            if dominated:
                k += 1
                if k >= x:
                    frontiers.append([index])
                    break
            else:
                frontiers[k].append(index)
                break

    return frontiers