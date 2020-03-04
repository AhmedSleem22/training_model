'''
In order to run the dataset_organization.py code, you have to write something similar to the following example in the OS Terminal:

$ python dataset_organization.py --input_dir ./dataset --left_dir ./dataset/Left --right_dir ./dataset/Right --forword_dir ./dataset/Forword

where:

    1. input_dir is the directory of the dataset folder.

    2. left_dir is the directory of the dataset left folder.

    3. right_dir is the directory of the dataset right folder.

    4. forword_dir is the directory of the dataset forword folder.
'''
import shutil, os
import argparse
import numpy as np
import csv

def main(args):
    # Load training data set from CSV file
    with open(os.path.join(args.input_dir, "data_file.csv"), 'r') as read_obj:
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


    for l in left:
        shutil.move(os.path.join(args.input_dir, l), args.left_dir)

    for f in forword:
        shutil.move(os.path.join(args.input_dir, f), args.forword_dir)

    for r in right:
        shutil.move(os.path.join(args.input_dir, r), args.right_dir)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='./dataset',
                        help='path to the directory of the dataset folder')
    parser.add_argument('--left_dir', type=str, default='./dataset/Left',
                        help='path to the directory where the left images will be saved to')
    parser.add_argument('--forword_dir', type=str, default='./dataset/Forword',
                        help='path to the directory where the forword images will be saved to')
    parser.add_argument('--right_dir', type=str, default='./dataset/Right',
                        help='path to the directory where the right images will be saved to')

    args = parser.parse_args()
    main(args)
