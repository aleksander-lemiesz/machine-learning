import argparse
import sys


def setup_input(x_val):
    i = 0
    n = 0
    k = 0
    # Setup input
    for line in sys.stdin:
        if line == "\n":
            break
        else:
            if i != 0:
                nums = line.split()
                in_nums = []
                for num in nums:
                    in_nums.append(float(num))
                x_val.append(in_nums)
            else:
                params = line.split()
                n = params[0]
                k = params[1]
            i += 1
    return n, k


def setup_train_file(train_file_name):
    train_file = open(train_file_name)
    train_x = []
    train_y = []
    for line in train_file:
        numbers = line.split()
        x_val = []
        for i in range(len(numbers) - 1):
            x_val.append(float(numbers[i]))
        train_x.append(x_val)
        train_y.append(float(numbers[len(numbers) - 1]))
    return train_x, train_y


def setup_in_file(train_in_name):
    in_file = open(train_in_name)
    line_num = []
    for line in in_file:
        line_num = line.split("=")
    in_file.close()
    return int(line_num[1])


def save_out_file(train_out_name, iters):
    out_file = open(train_out_name, "w")
    out_file.write('iterations=' + str(iters))
    out_file.close()


def stop_grad(grads, stop_diff):
    for grad in grads:
        if abs(grad) > stop_diff:
            return False
    return True


def print_polyn(n, k, polyn):
    print(str(n) + ' ' + str(k))
    for i in range(len(polyn)):
        nums = ''
        for j in range(len(polyn[i])):
            if j != len(polyn[i]) - 1:
                nums += str(int(polyn[i][j])) + ' '
            else:
                nums += str(polyn[i][j])
        print(nums)


def poly_calc_val(polyn, x_row):
    funcVal = 0
    # Iterate through described polynomial parts
    for part in polyn:
        partVal = 1
        factor = part[-1]
        rest = part[:-1]
        # For each variable in polynomial
        for x in rest:
            if x != 0:
                partVal *= x_row[int(x) - 1]
        funcVal += factor * partVal
    return funcVal


def grad_calc(polyn, p, funcVal, x_row, y):
    x_ind_row = polyn[p]
    grad = 0
    for x_ind in x_ind_row:
        if int(x_ind) == 0:
            grad += funcVal - y
        else:
            grad += (funcVal - y) * x_row[int(x_ind) - 1]
    return grad


if __name__ == '__main__':
    # Setup parameter
    parser = argparse.ArgumentParser(description='linTrain')
    parser.add_argument('-t', metavar='N', type=str, dest='train_set')
    parser.add_argument('-i', metavar='N', type=str, dest='data_in')
    parser.add_argument('-o', metavar='N', type=str, dest='data_out')
    args = parser.parse_args()
    # Setup and convert variables
    train_x, train_y = setup_train_file(args.train_set)
    max_iterations = setup_in_file(args.data_in)
    # Read input
    polyn_desc = []
    n, k = setup_input(polyn_desc)
    # Calculate
    N = len(train_y)
    alfa = 0.1
    stop_diff = 0.01
    grads = [1.0] * len(polyn_desc)

    # Iterate to learn
    for iterations in range(max_iterations):
        # For each of input data
        for i in range(len(polyn_desc)):
            grad = 0
            for train_point in range(N):
                funcVal = poly_calc_val(polyn_desc, train_x[train_point])
                grad += grad_calc(polyn_desc, i, funcVal, train_x[train_point], train_y[train_point])
            grads[i] = grad
            polyn_desc[i][-1] -= alfa * grad / N
        if stop_grad(grads, stop_diff):
            break
    # Print result
    print_polyn(n, k, polyn_desc)
    save_out_file(args.data_out, iterations + 1)


