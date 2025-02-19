import numpy as np
import random

def sampling_method(sampling_method_id, received_images_reshaped, received_labels_decoded, sample_size):
    # print("Percentage of filtering in our training dataset was set:")
    # print(sample_size)

    if sampling_method_id == 0:
        # Simple reservoir sampling over the whole training dataset
        # Total size of the stream (or training dataset)
        n_train = len(received_images_reshaped)

        # Number of samples that will be drawn
        k_train = int(n_train * sample_size)

        # Use the indexes of dataset in order to decide which samples will be drawn
        idx_tmp_train_list = list(range(0, n_train))

        # Find the indexes in order to construct the dataset that will be used during the training process
        idx_train = reservoir_sampling(idx_tmp_train_list, n_train, k_train)
    else:
        # Reservoir sampling in each class based on the number of samples (per class) that exist in the initial dataset
        # Find the size of each reservoir for every class depending on its occurence in the initial training dataset
        class_perc, unique_ids = reservoir_size_per_class(received_labels_decoded)
        # print(class_perc)
        # Stores the indexes (from all classes) in order to construct the dataset that will be used during the
        # training process
        idx_train = []

        # Run for every single class the reservoir sampling seperately
        for i in range(0, len(unique_ids)):
            # Find the locations of each sample belonging to our class of interest
            tmp = np.where(np.asarray(received_labels_decoded) == unique_ids[i])
            idx_of_class = tmp[0].tolist()

            # Run the reservoir sampling for the class of interest
            sampled_idx_of_class = reservoir_sampling(idx_of_class, len(idx_of_class),
                                                      int(len(received_images_reshaped) * sample_size * class_perc[i]))

            # Store the (sampled) samples from this class
            for j in range(0, len(sampled_idx_of_class)):
                idx_train.append(sampled_idx_of_class[j])

    # Store the corresponding images and labels from training dataset based on the sampled indexes
    train_images_lst = []
    for i in idx_train:
        train_images_lst.append(received_images_reshaped[i])

    train_labels_lst = []
    for i in idx_train:
        train_labels_lst.append(received_labels_decoded[i])

    # Check the occurrence of each class in the final training dataset
    # print_times_per_label(train_labels_lst, received_labels_decoded)

    # Transform the lists that we stored our samples into arrays
    train_images = np.asarray(train_images_lst)
    train_labels = np.asarray(train_labels_lst)

    # Verify that the desired filtering was performed in both datasets
    # print("Training dataset before sampling:")
    # print(len(received_images_reshaped))
    # print(len(received_labels_decoded))
    # print("Training dataset after sampling:")
    # print(train_images.shape)
    # print(train_labels.shape)

    return train_images, train_labels


# A function that finds the size of each reservoir for every class depending on its occurence in the initial dataset
# and returns the unique labels that exist in our dataset along with the corresponding percentage
def reservoir_size_per_class(init_labels):
    # Get unique labels and their counts (how many times they appear) in our training dataset
    unique_labels, counts = np.unique(init_labels, return_counts=True)

    # Transform to list
    unique_labels_lst = unique_labels.tolist()
    counts_lst = counts.tolist()

    perc_per_class = []
    for i in range(len(unique_labels_lst)):
        perc_per_class.append(counts_lst[i] / len(init_labels))

    # print(perc_per_class)

    return perc_per_class, unique_labels_lst


# Select k items from a stream of items-data

# A function to randomly select k items from stream[0..n-1].
def reservoir_sampling(stream, n, k):
    i = 0  # index for elements in stream[]

    # reservoir[] is the output array.
    # Initialize it with first k elements from stream[]
    reservoir = [0] * k

    for i in range(k):
        reservoir[i] = stream[i]

    # Iterate from the (k+1)th element to Nth element
    while (i < n):
        # Pick a random index from 0 to i.
        j = random.randrange(i + 1)

        # If the randomly picked
        # index is smaller than k,
        # then replace the element
        # present at the index
        # with new element from stream
        if (j < k):
            reservoir[j] = stream[i]
        i += 1
    return reservoir


# A function that prints the occurence of each class in a list
def print_times_per_label(lst, labels_all):
    # Get unique labels in our training dataset
    unique_labels = np.unique(labels_all)
    for i in range(0, len(unique_labels)):
        print("Class", unique_labels[i], "has", lst.count(i), "samples in our dataset...")
