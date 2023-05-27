import argparse
import random
import sys
from itertools import combinations_with_replacement
import math


def setup_train_file(train_file):
    file = open(train_file)
    train_x = []
    train_y = []
    for line in file:
        numbers = line.split()
        x_val = []
        for i in range(len(numbers) - 1):
            x_val.append(float(numbers[i]))
        train_x.append(x_val)
        train_y.append(float(numbers[len(numbers) - 1]))
    return train_x, train_y


def read_input():
    in_x = []
    for line in sys.stdin:
        if line == "\n":
            break
        else:
            nums = []
            xs = line.split()
            for x in xs:
                nums.append(float(x))
            in_x.append(nums)
    return in_x


def print_out(out_y):
    for y in out_y:
        print(y)


def setup():
    # Setup parameter
    parser = argparse.ArgumentParser(description='regressor')
    parser.add_argument('-t', metavar='N', type=str, dest='train_set')
    args = parser.parse_args()
    train_x, train_y = setup_train_file(args.train_set)
    in_x = read_input()
    out_y = [1.0] * len(in_x)
    return train_x, train_y, in_x, out_y


def poly_calc_val(polyn, x_row):
    funcVal = 0.0
    # Iterate through described polynomial parts
    for part in polyn:
        partVal = 1.0
        factor = part[-1]
        rest = part[:-1]
        # For each variable in polynomial
        for x in rest:
            if x != 0:
                partVal *= x_row[int(x) - 1]
        funcVal += factor * partVal
    return funcVal


def grad_calc(polyn, p, funcVal, x_row, y):
    diff = funcVal - y
    x_ind_row = polyn[p][:-1]
    for x_ind in x_ind_row:
        if int(x_ind) != 0:
            diff *= x_row[int(x_ind) - 1]
    if p != 0:
        diff += 0.001 * polyn[p][-1]
    return diff


def stop_grad(grads, stop_diff):
    for grad in grads:
        if abs(grad) > stop_diff:
            return False
    return True


def make_polyn(k, n):
    polyn_desc = []
    for row in list(combinations_with_replacement(range(n + 1), k)):
        polyn_desc.append(list(row))
    for i in range(len(polyn_desc)):
        polyn_desc[i].append(0.0)
    return polyn_desc


def normalize(y_val):
    transformed = []
    v_min = min(y_val)
    v_max = max(y_val)
    for v in y_val:
        trans = (2 * (v - v_min) / (v_max - v_min)) - 1
        transformed.append(trans)
    return transformed, v_min, v_max


def normalize_in(y_val, v_min, v_max):
    transformed = []
    for v in y_val:
        trans = (2 * (v - v_min) / (v_max - v_min)) - 1
        transformed.append(trans)
    return transformed


def return_val(trans_val, v_min, v_max):
    orig = []
    for trans in trans_val:
        orig.append((trans + 1) * ((v_max - v_min) / 2) + v_min)
    return orig


def trans_x(train_x):
    trans = []
    cols = []
    mins = []
    maxs = []
    for i in range(len(train_x[0])):
        col = [row[i] for row in train_x]
        col, v_min, v_max = normalize(col)
        mins.append(v_min)
        maxs.append(v_max)
        cols.append(col)
    for i in range(len(train_x)):
        row = [col[i] for col in cols]
        trans.append(row)
    #print(train_x)
    #print(trans)
    return trans, mins, maxs


def trans_x_in(train_x, mins, maxs):
    trans = []
    cols = []
    for i in range(len(train_x[0])):
        col = [row[i] for row in train_x]
        col = normalize_in(col, mins[i], maxs[i])
        cols.append(col)
    for i in range(len(train_x)):
        row = [col[i] for col in cols]
        trans.append(row)
    return trans


def trans_x_in_back(train_x, mins, maxs):
    trans = []
    cols = []
    for i in range(len(train_x[0])):
        col = [row[i] for row in train_x]
        col = return_val(col, mins[i], maxs[i])
        cols.append(col)
    for i in range(len(train_x)):
        row = [col[i] for col in cols]
        trans.append(row)
    return trans


def check_ys_norm(n_train_y):
    ###### Check if ys are normalized correctly ######
    print(n_train_y)
    n_train_y_t, n_min, n_max = normalize(n_train_y)
    print(n_train_y_t)
    print(return_val(n_train_y_t, n_min, n_max))


def check_in_norm(in_x, t_in_x, mins, maxs):
    ###### Check if input is normalized correctly ######
    print(in_x)
    print(t_in_x)
    print(trans_x_in_back(t_in_x, mins, maxs))


def learn(polyn_desc, grads, train_x, train_y, alfa, iters, N, stop_diff):
    # Iterate to learn
    for iterations in range(iters):
        # For each of input data
        for i in range(len(polyn_desc)):
            grad = 0
            for train_point in range(N):
                funcVal = poly_calc_val(polyn_desc, train_x[train_point])
                grad += grad_calc(polyn_desc, i, funcVal, train_x[train_point], train_y[train_point])
            if math.isnan(grad) or math.isinf(grad):
                raise OverflowError
            grads[i] = grad
            # print(polyn_desc)
        for i in range(len(polyn_desc)):
            polyn_desc[i][-1] -= alfa * grads[i] / N
        if stop_grad(grads, stop_diff):
            break


def split_data(train_data):
    train_x = []
    train_y = []
    for t in train_data:
        y = t[-1]
        xs = t[:-1]
        train_x.append(xs)
        train_y.append(y)
    return train_x, train_y


def merge_data(train_x, train_y):
    train_data_transformed = []
    for i in range(len(train_x)):
        trans_data = train_x[i]
        trans_data.append(train_y[i])
        train_data_transformed.append(trans_data)
    return train_data_transformed


def validate(n, train_x, train_y):
    N = len(train_y)
    q_min = 99999999999999999999.9
    max_k = 15
    k_min = max_k

    for s in range(max_k):
        # print(s)
        polyn_desc = make_polyn(s + 1, n)
        grads = [1.0] * len(polyn_desc)
        alfa = 0.4
        stop_diff = 0.01

        learn(polyn_desc, grads, train_x, train_y, alfa, 1000, N, stop_diff)

        q = 0
        for i, x_row in enumerate(train_x):
            q += (poly_calc_val(polyn_desc, x_row) - train_y[i])**2
        if q < q_min:
            q_min = q
            k_min = s + 1

    return k_min


def p_validate(n, train_x, train_y):
    if n == 1:
        k = 4
        if any(x[0] <= -3 for x in train_x) and all(y >= -3 for y in train_y):
            alfa = 10 ** -4
    elif n == 3:
        k = 2
    elif n == 2:
        k = 3
    else:
        k = 1
    return k


def validate_v2(n, train_x, train_y):
    q_min = 99999999999999999999.9
    max_k = 15
    k_min = max_k

    merged_data = merge_data(train_x, train_y)

    validation_ratio = 0.3
    selected_train = []
    selected_val = []
    data_to_select_initial = [row for row in merged_data]
    no_shuffles = 4

    validation_set_size = int(math.floor((validation_ratio * len(data_to_select_initial))))
    for i in range(no_shuffles):
        random.shuffle(data_to_select_initial)
        validation_set = data_to_select_initial[:validation_set_size]
        train_set = data_to_select_initial[validation_set_size:]
        selected_val.append(validation_set)
        selected_train.append(train_set)

    for s in range(max_k):
        q = 0.0
        cur_k = s + 1
        for i in range(no_shuffles):
            data_train = selected_train[i]
            data_val = selected_val[i]

            polyn_desc = make_polyn(cur_k, n)
            grads = [1.0] * len(polyn_desc)
            alfa = 0.4
            stop_diff = 0.01

            sel_t_x, sel_t_y = split_data(data_train)
            N = len(sel_t_y)

            learn(polyn_desc, grads, sel_t_x, sel_t_y, alfa, 1000, N, stop_diff)

            val_x, val_y = split_data(data_val)
            for j, x_row in enumerate(val_x):
                q += (poly_calc_val(polyn_desc, x_row) - val_y[j])**2

        q /= no_shuffles
        if q < q_min:
            q_min = q
            k_min = cur_k

    return k_min


def reg():
    train_x, train_y, in_x, out_y = setup()
    # check_ys_norm(train_y.copy())
    train_x, mins, maxs = trans_x(train_x)
    train_y, y_min, y_max = normalize(train_y)

    n = len(train_x[0])
    # k = p_validate(n, train_x, train_y)
    # k = validate(n, train_x, train_y)
    k = validate_v2(n, train_x, train_y)
    # k = 4

    print(k)
    print()
    polyn_desc = make_polyn(k, n)

    N = len(train_y)
    alfa = 0.4
    stop_diff = 0.00001
    grads = [1.0] * len(polyn_desc)

    learn(polyn_desc, grads, train_x, train_y, alfa, 180000, N, stop_diff)

    res = []
    t_in_x = trans_x_in(in_x, mins, maxs)

    # check_in_norm(in_x, t_in_x, mins, maxs)

    for xs in t_in_x:
        res.append(poly_calc_val(polyn_desc, xs))
    res = return_val(res, y_min, y_max)
    print_out(res)


def test():
    orig = [1, 6, 7, 9, 5, 30]
    trans = normalize(orig)
    rev = return_val(trans[0], trans[1], trans[2])
    print(trans)
    print(rev)
    print(orig)


if __name__ == '__main__':
    reg()
    # test()


