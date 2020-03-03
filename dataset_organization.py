import shutil, os
import argparse
import numpy as np
import csv

def main(args):
    # Load training data set from CSV file
    #training_data_df = pd.read_csv("data_file.csv", sep=',')

    # Pull out columns for X (data to train with) and Y (value to predict)
    with open("data_file.csv", 'r') as read_obj:
        csv_reader = csv.DictReader(read_obj, delimiter=',')

        right  = []
        left   = []
        forword= []
        for line in csv_reader:

            if ([int(line['Right']), int(line['Forword']), int(line['Left'])] == [0, 1, 1] or
                    [int(line['Right']), int(line['Forword']), int(line['Left'])] == [0, 0, 1]):
                left.append(line['Images'])

            elif ([int(line['Right']), int(line['Forword']), int(line['Left'])] == [1, 0, 1] or
                    [int(line['Right']), int(line['Forword']), int(line['Left'])] == [1, 1, 1]):
                forword.append(line['Images'])

            elif ([int(line['Right']), int(line['Forword']), int(line['Left'])] == [1, 1, 0] or
                    [int(line['Right']), int(line['Forword']), int(line['Left'])] == [1, 0, 0]):
                right.append(line['Images'])

    print(left)
    print(forword)
    print(right)

    for l in left:
        shutil.move(l, args.left_dir)

    for f in forword:
        shutil.move(f, args.forword_dir)

    for r in right:
        shutil.move(r, args.right_dir)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--left_dir', type=str, default='Left',
                        help='path to the directory where the left images will be saved to')
    parser.add_argument('--forword_dir', type=str, default='Forword',
                        help='path to the directory where the forword images will be saved to')
    parser.add_argument('--right_dir', type=str, default='Right',
                        help='path to the directory where the right images will be saved to')

    args = parser.parse_args()
    main(args)
