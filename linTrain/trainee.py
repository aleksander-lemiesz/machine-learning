import argparse
import sys

if __name__ == '__main__':
    # Setup parameter
    parser = argparse.ArgumentParser(description='Polynomial')
    parser.add_argument('-d', metavar='N', type=str, dest='description')
    args = parser.parse_args()
    # Setup and convert variables from description.txt
    descFile = open(args.description)
    description = []
    for i, line in enumerate(descFile):
        if i == 0:
            continue
        lineNum = line.split()
        nums = []
        for number in lineNum:
            nums.append(float(number))
        description.append(nums)
    descFile.close()
    # Setup input
    xValues = []
    for line in sys.stdin:
        if line == "\n":
            break
        else:
            nums = line.split()
            inNums = []
            for num in nums:
                inNums.append(float(num))
            xValues.append(inNums)
    # Calculate

    # For each of input data
    for ex in xValues:
        funcVal = 0
        # Iterate through described polynomial parts
        for part in description:
            partVal = 1
            factor = part[-1]
            rest = part[:-1]
            # For each variable in polynomial
            for x in rest:
                if x != 0:
                    partVal *= ex[int(x) - 1]
            funcVal += factor * partVal
        print(funcVal)
