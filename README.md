# training_model
We collected the data in the form of separate images and with it a CSV file containing the names of the images and the road the car took.

**Stage 1:**

The first step we take is to organize the data by **dataset\_organization.py** code

Where we separate the images in which the car was going to the right in a separate file, as well as the left and the front,and grouping all the folders in one folder &quot;dataset&quot;.

In order to run the **dataset\_organization.py** code, you have to write something similar to the following example in the OS Terminal:

**$ python dataset\_organization.py --input\_dir ./dataset --left\_dir ./dataset/Left --right\_dir ./dataset/Right --forword\_dir ./dataset/Forword**

where:

        1. input_dir is the directory of the dataset folder.

        2. left_dir is the directory of the dataset left folder.

        3. right_dir is the directory of the dataset right folder.

        4. forword_dir is the directory of the dataset forword folder.

**Stage 2:**

The second step we take is to create TFRecords from the data by **create\_tf\_records.py** Code

so that we can enter it into the training code.

In order to run the **create\_tf\_records.py** code, you have to write something similar to the following example in the OS Terminal:

**$ python create\_tf\_records.py --input\_dir ./dataset --output\_dir ./tfrecords --num\_shards 10 --split\_ratio 0.2**

where:

        1. input_dir is the directory of the dataset folder.

        2. output_dir is the directory of the output tfrecords.

        3. num_shards is the number of the tfrecords files that you want to split the data on.

        4. split_ratio is the number of testing files divided into the total number of files.

**Stage 3:**

The third step we take is to run the **trainer.py** code.

In order to run the **trainer.py** code, you have to write something similar to the following example in the OS Terminal:

**$ python trainer.py --checkpoint\_path ./checkpoints --data\_path ./tfrecords**

where:

        1. checkpoint_path is the directory of the saver folder.

        2. data_path is the directory of the tfrecords folder.
