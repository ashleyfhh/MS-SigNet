import os
import natsort
import random
import numpy as np
from config import arg_conf


args = arg_conf()
np.random.seed(args.random_seed)

# Build a DatasetName class for all datasets
class DatasetName:
    pass

def data_path(DatasetName):
    global total_genuine, total_forged
    main_path = DatasetName
    genuine_path = os.path.join(main_path, 'genuine_otsu/')
    forged_path = os.path.join(main_path, 'forged_otsu/')

    if main_path == 'CEDAR':
        total_genuine = natsort.natsorted(os.listdir(genuine_path))
        total_forged = natsort.natsorted(os.listdir(forged_path))
    elif main_path == 'HanSig':
        total_genuine = natsort.natsorted(os.listdir(genuine_path))
        total_forged = natsort.natsorted(os.listdir(forged_path))
    elif main_path == 'BHSig_B':
        total_genuine = []
        total_forged = []
        for i in range(1, 101):
            total_genuine+= os.listdir(os.path.join(genuine_path, str(i).zfill(3)))
            total_forged+= os.listdir(os.path.join(forged_path, str(i).zfill(3)))
        total_genuine = natsort.natsorted(total_genuine)
        total_forged = natsort.natsorted(total_forged)
    elif main_path == 'BHSig_H':
        total_genuine = []
        total_forged = []
        for i in range(1, 161):
            total_genuine+= os.listdir(os.path.join(genuine_path, str(i).zfill(3)))
            total_forged+= os.listdir(os.path.join(forged_path, str(i).zfill(3)))
        total_genuine = natsort.natsorted(total_genuine)
        total_forged = natsort.natsorted(total_forged)

    return genuine_path, forged_path, total_genuine, total_forged

# Split training/validation/test subset (for CEDAR & BHSig260 dataset, one writer singned only one set of names)
def data_split_1(num_writers, num_train_writer, num_valid_writer, filenames, seed=args.data_seed):
    train_set = []
    valid_set = []
    test_set = []

    # Set random seed (genuine and forged signatures shoud belong to the same writers)
    random.seed(seed)

    # Random select the writers for training/valid/test set
    all_writers = list(range(1, num_writers+1))
    random.shuffle(all_writers)
    train_writers = all_writers[0:num_train_writer]
    valid_writers = all_writers[num_train_writer:num_train_writer+num_valid_writer]

    # Get files for training/valid/test set
    for file in filenames:
        if int(file.split('_')[1]) in train_writers:
            train_set.append(file)
        elif int(file.split('_')[1]) in valid_writers:
            valid_set.append(file)
        else:
            test_set.append(file)

    # print(len(train_set))
    # print(len(valid_set))
    # print(len(test_set))

    return train_set, valid_set, test_set

# Split training/validation/test subset (for HanSig dataset, one writer singned more than one set of names)
def data_split_2(num_writers, num_train_writer, num_valid_writer, filenames, seed=args.data_seed):
    train_set = []
    valid_set = []
    test_set = []

    # Set random seed (genuine and forged signatures shoud belong to the same writers)
    random.seed(seed)

    # Random select the writers for training/valid/test set
    all_writers = list(range(1, num_writers+1))
    random.shuffle(all_writers)
    train_writers = all_writers[0:num_train_writer]
    valid_writers = all_writers[num_train_writer:num_train_writer+num_valid_writer]

    # Get files for training/valid/test set
    for file in filenames:
        if int(file.split('_')[1].strip('w')) in train_writers:
            train_set.append(file)
        elif int(file.split('_')[1].strip('w')) in valid_writers:
            valid_set.append(file)
        else:
            test_set.append(file)

    # print(len(train_set))
    # print(len(valid_set))
    # print(len(test_set))

    return train_set, valid_set, test_set


# -- CEDAR (55 writers, each has 24 signatures)
CEDAR = DatasetName()
# Dataset path
CEDAR.path_genuine, CEDAR.path_forged, CEDAR.files_genuine, CEDAR.files_forged = data_path('CEDAR')

# Set training set/valid set/test set
# Random selection from writers
CEDAR.train_genuine, CEDAR.valid_genuine, CEDAR.test_genuine = data_split_1(55, 45, 5, CEDAR.files_genuine)
CEDAR.train_forged, CEDAR.valid_forged, CEDAR.test_forged = data_split_1(55, 45, 5, CEDAR.files_forged)

# Indices of each writer's images in the training set
CEDAR.train_writers = np.array([i.split('_')[1] for i in CEDAR.train_genuine], dtype=np.float64)
CEDAR.train_writer_set = set(CEDAR.train_writers)
# writer_indices: {1: array([0, 1,..., 23], dtype=int64), 2: array([24, 25,..., 47], dtype=int64), ...}
CEDAR.train_writer_indices = {label: np.where(CEDAR.train_writers == label)[0]
                              for label in CEDAR.train_writer_set}

# Indices of each writer's images in the validation set
CEDAR.valid_writers = np.array([i.split('_')[1] for i in CEDAR.valid_genuine], dtype=np.float64)
CEDAR.valid_writer_set = set(CEDAR.valid_writers)
# writer_indices
CEDAR.valid_writer_indices = {label: np.where(CEDAR.valid_writers == label)[0]
                             for label in CEDAR.valid_writer_set}

# Indices of each writer's images in the test set
CEDAR.test_writers = np.array([i.split('_')[1] for i in CEDAR.test_genuine], dtype=np.float64)
CEDAR.test_writer_set = set(CEDAR.test_writers)
# writer_indices
CEDAR.test_writer_indices = {label: np.where(CEDAR.test_writers == label)[0]
                             for label in CEDAR.test_writer_set}

# -- Generate validation pairs (similar & dissimilar pairs) and corresponding pair labels (y=1 & y=0)
CEDAR.valid_pair_indices = []
CEDAR.valid_pair_labels = []
# for each writer(w), generate valid pairs of #ref=1
for w in CEDAR.valid_writer_set:
    similar_pairs = []
    similar_labels = []
    dissimilar_pairs = []
    dissimilar_labels = []

    img_idx_each_writer = CEDAR.valid_writer_indices[w]
    # for each writer(w), select one genuine as reference, another one genuine as positive test image
    for i in range(0, len(img_idx_each_writer) - 1):
        for j in range(i+1, len(img_idx_each_writer)):
            similar_pairs += [[img_idx_each_writer[i], img_idx_each_writer[j]]]
            similar_labels += [1]
    # for each writer(w), select one genuine as reference, and select one forged as negative test image
    for k in range(0, len(img_idx_each_writer)):
        for m in range(0, len(img_idx_each_writer)):
            dissimilar_pairs += [[img_idx_each_writer[k], img_idx_each_writer[m]]]
            dissimilar_labels += [0]
    # combine "pairs and labels of a writer" to the list
    CEDAR.valid_pair_indices += similar_pairs
    CEDAR.valid_pair_labels += similar_labels
    CEDAR.valid_pair_indices += dissimilar_pairs
    CEDAR.valid_pair_labels += dissimilar_labels

# Convert list to array
CEDAR.valid_pair_indices = np.array(CEDAR.valid_pair_indices)
CEDAR.valid_pair_labels = np.array(CEDAR.valid_pair_labels)
print('(CEDAR) Number of valid pairs:', len(CEDAR.valid_pair_indices))
print('(CEDAR) Number of valid labels:', len(CEDAR.valid_pair_labels))


# -- Generate test pairs (similar & dissimilar pairs) and corresponding pair labels (y=1 & y=0)
CEDAR.test_pair_indices = []
CEDAR.test_pair_labels = []
# for each writer(w), generate test pairs of #ref=1
for w in CEDAR.test_writer_set:
    similar_pairs = []
    similar_labels = []
    dissimilar_pairs = []
    dissimilar_labels = []

    img_idx_each_writer = CEDAR.test_writer_indices[w]
    # for each writer(w), select one genuine as reference, another one genuine as positive test image
    for i in range(0, len(img_idx_each_writer) - 1):
        for j in range(i+1, len(img_idx_each_writer)):
            similar_pairs += [[img_idx_each_writer[i], img_idx_each_writer[j]]]
            similar_labels += [1]
    # for each writer(w), select one genuine as reference, and select one forged as negative test image
    for k in range(0, len(img_idx_each_writer)):
        for m in range(0, len(img_idx_each_writer)):
            dissimilar_pairs += [[img_idx_each_writer[k], img_idx_each_writer[m]]]
            dissimilar_labels += [0]
    # If we randomly select 276 dissimilar pairs without replacement
    random.seed(args.data_seed)
    dissimilar_pairs = random.sample(dissimilar_pairs, k=276)
    dissimilar_labels = dissimilar_labels[0:276]
    # combine "pairs and labels of a writer" to the list
    CEDAR.test_pair_indices += similar_pairs
    CEDAR.test_pair_labels += similar_labels
    CEDAR.test_pair_indices += dissimilar_pairs
    CEDAR.test_pair_labels += dissimilar_labels

# Convert list to array
CEDAR.test_pair_indices = np.array(CEDAR.test_pair_indices)
CEDAR.test_pair_labels = np.array(CEDAR.test_pair_labels)
print('(CEDAR) Number of test pairs:', len(CEDAR.test_pair_indices))
print('(CEDAR) Number of test labels:', len(CEDAR.test_pair_labels))

# -- HanSig (238 writers, each has 60 or 120 genuine signatures => genuine 17,700 & forged 17,700)
HanSig = DatasetName()
# Dataset path
HanSig.path_genuine, HanSig.path_forged, HanSig.files_genuine, HanSig.files_forged = data_path('HanSig')

# Set training set/valid set/test set
# Random selection from writers
HanSig.train_genuine, HanSig.valid_genuine, HanSig.test_genuine = data_split_2(238, 198, 20, HanSig.files_genuine)
HanSig.train_forged, HanSig.valid_forged, HanSig.test_forged = data_split_2(238, 198, 20, HanSig.files_forged)

# Indices of each name's images in the training set
# Each writer has signed more than one names, so we practically use the name index here  ("2" in "original_w1_2_15.jpg")
HanSig.train_writers = np.array([i.split('_')[2] for i in HanSig.train_genuine], dtype=np.float64)
HanSig.train_writer_set = set(HanSig.train_writers)
# writer_indices: {1: array([0, 1,..., 19], dtype=int64), 2: array([20, 21,..., 39], dtype=int64), ...}
HanSig.train_writer_indices = {label: np.where(HanSig.train_writers == label)[0]
                               for label in HanSig.train_writer_set}

# Indices of each name's images in the validation set
HanSig.valid_writers = np.array([i.split('_')[2] for i in HanSig.valid_genuine], dtype=np.float64)
HanSig.valid_writer_set = set(HanSig.valid_writers)
# writer_indices
HanSig.valid_writer_indices = {label: np.where(HanSig.valid_writers == label)[0]
                               for label in HanSig.valid_writer_set}

# Indices of each name's images in the test set
HanSig.test_writers = np.array([i.split('_')[2] for i in HanSig.test_genuine], dtype=np.float64)
HanSig.test_writer_set = set(HanSig.test_writers)
# writer_indices
HanSig.test_writer_indices = {label: np.where(HanSig.test_writers == label)[0]
                              for label in HanSig.test_writer_set}

# -- Generate validation pairs (similar & dissimilar pairs) and corresponding pair labels (y=1 & y=0)
HanSig.valid_pair_indices = []
HanSig.valid_pair_labels = []
# for each writer(w), generate valid pairs of #ref=1
# Practically for each "name", not for each writer here
for w in HanSig.valid_writer_set:
    similar_pairs = []
    similar_labels = []
    dissimilar_pairs = []
    dissimilar_labels = []

    img_idx_each_writer = HanSig.valid_writer_indices[w]
    # for each writer(w), select one genuine as reference, another one genuine as positive test image
    for i in range(0, len(img_idx_each_writer) - 1):
        for j in range(i+1, len(img_idx_each_writer)):
            similar_pairs += [[img_idx_each_writer[i], img_idx_each_writer[j]]]
            similar_labels += [1]
    # for each writer(w), select one genuine as reference, and select one forged as negative test image
    for k in range(0, len(img_idx_each_writer)):
        for m in range(0, len(img_idx_each_writer)):
            dissimilar_pairs += [[img_idx_each_writer[k], img_idx_each_writer[m]]]
            dissimilar_labels += [0]
    # combine "pairs and labels of a writer (of a name practically)" to the list
    HanSig.valid_pair_indices += similar_pairs
    HanSig.valid_pair_labels += similar_labels
    HanSig.valid_pair_indices += dissimilar_pairs
    HanSig.valid_pair_labels += dissimilar_labels

# Convert list to array
HanSig.valid_pair_indices = np.array(HanSig.valid_pair_indices)
HanSig.valid_pair_labels = np.array(HanSig.valid_pair_labels)
print('(HanSig) Number of valid pairs:', len(HanSig.valid_pair_indices))
print('(HanSig) Number of valid labels:', len(HanSig.valid_pair_labels))

# -- Generate test pairs (similar & dissimilar pairs) and corresponding pair labels (y=1 & y=0)
HanSig.test_pair_indices = []
HanSig.test_pair_labels = []
# for each writer(w), generate test pairs of #ref=1
# Practically for each "name", not for each writer here
for w in HanSig.test_writer_set:
    similar_pairs = []
    similar_labels = []
    dissimilar_pairs = []
    dissimilar_labels = []

    img_idx_each_writer = HanSig.test_writer_indices[w]
    # for each writer(w), select one genuine as reference, another one genuine as positive test image
    for i in range(0, len(img_idx_each_writer) - 1):
        for j in range(i+1, len(img_idx_each_writer)):
            similar_pairs += [[img_idx_each_writer[i], img_idx_each_writer[j]]]
            similar_labels += [1]
    # for each writer(w), select one genuine as reference, and select one forged as negative test image
    for k in range(0, len(img_idx_each_writer)):
        for m in range(0, len(img_idx_each_writer)):
            dissimilar_pairs += [[img_idx_each_writer[k], img_idx_each_writer[m]]]
            dissimilar_labels += [0]
    # If we randomly select 190 dissimilar pairs without replacement
    random.seed(args.data_seed)
    dissimilar_pairs = random.sample(dissimilar_pairs, k=190)
    dissimilar_labels = dissimilar_labels[0:190]
    # combine "pairs and labels of a writer (of a name practically)" to the list
    HanSig.test_pair_indices += similar_pairs
    HanSig.test_pair_labels += similar_labels
    HanSig.test_pair_indices += dissimilar_pairs
    HanSig.test_pair_labels += dissimilar_labels

# Convert list to array
HanSig.test_pair_indices = np.array(HanSig.test_pair_indices)
HanSig.test_pair_labels = np.array(HanSig.test_pair_labels)
print('(HanSig) Number of test pairs:', len(HanSig.test_pair_indices))
print('(HanSig) Number of test labels:', len(HanSig.test_pair_labels))


# -- BHSig-Bengali (100 writers, each has 24 genuine and 30 forged signatures)
# BHSig has different numbers of genuine and forged images => needs to change data codes
BHSig_B = DatasetName()
# Dataset path
BHSig_B.path_genuine, BHSig_B.path_forged, BHSig_B.files_genuine, BHSig_B.files_forged = data_path('BHSig_B')

# Set training set/valid set/test set
# Random selection from writers
BHSig_B.train_genuine, BHSig_B.valid_genuine, BHSig_B.test_genuine = data_split_1(100, 45, 5, BHSig_B.files_genuine)
BHSig_B.train_forged, BHSig_B.valid_forged, BHSig_B.test_forged = data_split_1(100, 45, 5, BHSig_B.files_forged)

# Indices of each writer's "genuine" images in the training set
BHSig_B.train_writers_gen = np.array([i.split('_')[1] for i in BHSig_B.train_genuine], dtype=np.float64)
# the same writer set for genuine and forged images
BHSig_B.train_writer_set = set(BHSig_B.train_writers_gen)
# writer_indices: {1: array([0, 1,..., 23], dtype=int64), 2: array([24, 25,..., 47], dtype=int64), ...}
BHSig_B.train_writer_indices_gen = {label: np.where(BHSig_B.train_writers_gen == label)[0]
                                    for label in BHSig_B.train_writer_set}
# Indices of each writer's "forged" images in the training set
BHSig_B.train_writers_fgd = np.array([i.split('_')[1] for i in BHSig_B.train_forged], dtype=np.float64)
# writer_indices of forged
BHSig_B.train_writer_indices_fgd = {label: np.where(BHSig_B.train_writers_fgd == label)[0]
                                    for label in BHSig_B.train_writer_set}

# Indices of each writer's "genuine" images in the validation set
BHSig_B.valid_writers_gen = np.array([i.split('_')[1] for i in BHSig_B.valid_genuine], dtype=np.float64)
# the same writer set for genuine and forged images
BHSig_B.valid_writer_set = set(BHSig_B.valid_writers_gen)
# writer_indices
BHSig_B.valid_writer_indices_gen = {label: np.where(BHSig_B.valid_writers_gen == label)[0]
                                    for label in BHSig_B.valid_writer_set}
# Indices of each writer's "forged" images in the validation set
BHSig_B.valid_writers_fgd = np.array([i.split('_')[1] for i in BHSig_B.valid_forged], dtype=np.float64)
# writer_indices of forged
BHSig_B.valid_writer_indices_fgd = {label: np.where(BHSig_B.valid_writers_fgd == label)[0]
                                    for label in BHSig_B.valid_writer_set}

# Indices of each writer's "genuine" images in the test set
BHSig_B.test_writers_gen = np.array([i.split('_')[1] for i in BHSig_B.test_genuine], dtype=np.float64)
# the same writer set for genuine and forged images
BHSig_B.test_writer_set = set(BHSig_B.test_writers_gen)
# writer_indices of genuine
BHSig_B.test_writer_indices_gen = {label: np.where(BHSig_B.test_writers_gen == label)[0]
                                   for label in BHSig_B.test_writer_set}
# Indices of each writer's "forged" images in the test set
BHSig_B.test_writers_fgd = np.array([i.split('_')[1] for i in BHSig_B.test_forged], dtype=np.float64)
# writer_indices of forged
BHSig_B.test_writer_indices_fgd = {label: np.where(BHSig_B.test_writers_fgd == label)[0]
                                   for label in BHSig_B.test_writer_set}

# -- Generate validation pairs (similar & dissimilar pairs) and corresponding pair labels (y=1 & y=0)
BHSig_B.valid_pair_indices = []
BHSig_B.valid_pair_labels = []
# for each writer(w), generate valid pairs of #ref=1
for w in BHSig_B.valid_writer_set:
    similar_pairs = []
    similar_labels = []
    dissimilar_pairs = []
    dissimilar_labels = []

    img_idx_each_writer_gen = BHSig_B.valid_writer_indices_gen[w]
    img_idx_each_writer_fgd = BHSig_B.valid_writer_indices_fgd[w]
    # for each writer(w), select one genuine as reference, another one genuine as positive test image
    for i in range(0, len(img_idx_each_writer_gen) - 1):
        for j in range(i+1, len(img_idx_each_writer_gen)):
            similar_pairs += [[img_idx_each_writer_gen[i], img_idx_each_writer_gen[j]]]
            similar_labels += [1]
    # for each writer(w), select one genuine as reference, and select one forged as negative test image
    for k in range(0, len(img_idx_each_writer_gen)):
        for m in range(0, len(img_idx_each_writer_fgd)):
            dissimilar_pairs += [[img_idx_each_writer_gen[k], img_idx_each_writer_fgd[m]]]
            dissimilar_labels += [0]
    # combine "pairs and labels of a writer" to the list
    BHSig_B.valid_pair_indices += similar_pairs
    BHSig_B.valid_pair_labels += similar_labels
    BHSig_B.valid_pair_indices += dissimilar_pairs
    BHSig_B.valid_pair_labels += dissimilar_labels

# Convert list to array
BHSig_B.valid_pair_indices = np.array(BHSig_B.valid_pair_indices)
BHSig_B.valid_pair_labels = np.array(BHSig_B.valid_pair_labels)
print('(BHSig_B) Number of valid pairs:', len(BHSig_B.valid_pair_indices))
print('(BHSig_B) Number of valid labels:', len(BHSig_B.valid_pair_labels))

# -- Generate test pairs (similar & dissimilar pairs) and corresponding pair labels (y=1 & y=0)
BHSig_B.test_pair_indices = []
BHSig_B.test_pair_labels = []
# for each writer(w), generate test pairs of #ref=1
for w in BHSig_B.test_writer_set:
    similar_pairs = []
    similar_labels = []
    dissimilar_pairs = []
    dissimilar_labels = []

    img_idx_each_writer_gen = BHSig_B.test_writer_indices_gen[w]
    img_idx_each_writer_fgd = BHSig_B.test_writer_indices_fgd[w]
    # for each writer(w), select one genuine as reference, another one genuine as positive test image
    for i in range(0, len(img_idx_each_writer_gen) - 1):
        for j in range(i+1, len(img_idx_each_writer_gen)):
            similar_pairs += [[img_idx_each_writer_gen[i], img_idx_each_writer_gen[j]]]
            similar_labels += [1]
    # for each writer(w), select one genuine as reference, and select one forged as negative test image
    for k in range(0, len(img_idx_each_writer_gen)):
        for m in range(0, len(img_idx_each_writer_fgd)):
            dissimilar_pairs += [[img_idx_each_writer_gen[k], img_idx_each_writer_fgd[m]]]
            dissimilar_labels += [0]
    # But we follow other papers to random select 276 dissimilar pairs without replacement
    random.seed(args.data_seed)
    dissimilar_pairs = random.sample(dissimilar_pairs, k=276)
    dissimilar_labels = dissimilar_labels[0:276]
    # combine "pairs and labels of a writer" to the list
    BHSig_B.test_pair_indices += similar_pairs
    BHSig_B.test_pair_labels += similar_labels
    BHSig_B.test_pair_indices += dissimilar_pairs
    BHSig_B.test_pair_labels += dissimilar_labels

# Convert list to array
BHSig_B.test_pair_indices = np.array(BHSig_B.test_pair_indices)
BHSig_B.test_pair_labels = np.array(BHSig_B.test_pair_labels)
print('(BHSig_B) Number of test pairs:', len(BHSig_B.test_pair_indices))
print('(BHSig_B) Number of test labels:', len(BHSig_B.test_pair_labels))

# -- BHSig-Hindi (160 writers, each has 24 genuine and 30 forged signatures)
# BHSig has different numbers of genuine and forged images => needs to change data codes
BHSig_H = DatasetName()
# Dataset path
BHSig_H.path_genuine, BHSig_H.path_forged, BHSig_H.files_genuine, BHSig_H.files_forged = data_path('BHSig_H')

# Set training set/valid set/test set
# Random selection from writers
BHSig_H.train_genuine, BHSig_H.valid_genuine, BHSig_H.test_genuine = data_split_1(160, 95, 5, BHSig_H.files_genuine)
BHSig_H.train_forged, BHSig_H.valid_forged, BHSig_H.test_forged = data_split_1(160, 95, 5, BHSig_H.files_forged)

# Indices of each writer's "genuine" images in the training set
BHSig_H.train_writers_gen = np.array([i.split('_')[1] for i in BHSig_H.train_genuine], dtype=np.float64)
# the same writer set for genuine and forged images
BHSig_H.train_writer_set = set(BHSig_H.train_writers_gen)
# writer_indices: {1: array([0, 1,..., 23], dtype=int64), 2: array([24, 25,..., 47], dtype=int64), ...}
BHSig_H.train_writer_indices_gen = {label: np.where(BHSig_H.train_writers_gen == label)[0]
                                    for label in BHSig_H.train_writer_set}
# Indices of each writer's "forged" images in the training set
BHSig_H.train_writers_fgd = np.array([i.split('_')[1] for i in BHSig_H.train_forged], dtype=np.float64)
# writer_indices of forged
BHSig_H.train_writer_indices_fgd = {label: np.where(BHSig_H.train_writers_fgd == label)[0]
                                    for label in BHSig_H.train_writer_set}

# Indices of each writer's "genuine" images in the validation set
BHSig_H.valid_writers_gen = np.array([i.split('_')[1] for i in BHSig_H.valid_genuine], dtype=np.float64)
# the same writer set for genuine and forged images
BHSig_H.valid_writer_set = set(BHSig_H.valid_writers_gen)
# writer_indices
BHSig_H.valid_writer_indices_gen = {label: np.where(BHSig_H.valid_writers_gen == label)[0]
                                    for label in BHSig_H.valid_writer_set}
# Indices of each writer's "forged" images in the validation set
BHSig_H.valid_writers_fgd = np.array([i.split('_')[1] for i in BHSig_H.valid_forged], dtype=np.float64)
# writer_indices of forged
BHSig_H.valid_writer_indices_fgd = {label: np.where(BHSig_H.valid_writers_fgd == label)[0]
                                    for label in BHSig_H.valid_writer_set}

# Indices of each writer's "genuine" images in the test set
BHSig_H.test_writers_gen = np.array([i.split('_')[1] for i in BHSig_H.test_genuine], dtype=np.float64)
# the same writer set for genuine and forged images
BHSig_H.test_writer_set = set(BHSig_H.test_writers_gen)
# writer_indices of genuine
BHSig_H.test_writer_indices_gen = {label: np.where(BHSig_H.test_writers_gen == label)[0]
                                   for label in BHSig_H.test_writer_set}
# Indices of each writer's "forged" images in the test set
BHSig_H.test_writers_fgd = np.array([i.split('_')[1] for i in BHSig_H.test_forged], dtype=np.float64)
# writer_indices of forged
BHSig_H.test_writer_indices_fgd = {label: np.where(BHSig_H.test_writers_fgd == label)[0]
                                   for label in BHSig_H.test_writer_set}

# -- Generate validation pairs (similar & dissimilar pairs) and corresponding pair labels (y=1 & y=0)
BHSig_H.valid_pair_indices = []
BHSig_H.valid_pair_labels = []
# for each writer(w), generate valid pairs of #ref=1
for w in BHSig_H.valid_writer_set:
    similar_pairs = []
    similar_labels = []
    dissimilar_pairs = []
    dissimilar_labels = []

    img_idx_each_writer_gen = BHSig_H.valid_writer_indices_gen[w]
    img_idx_each_writer_fgd = BHSig_H.valid_writer_indices_fgd[w]
    # for each writer(w), select one genuine as reference, another one genuine as positive test image
    for i in range(0, len(img_idx_each_writer_gen) - 1):
        for j in range(i+1, len(img_idx_each_writer_gen)):
            similar_pairs += [[img_idx_each_writer_gen[i], img_idx_each_writer_gen[j]]]
            similar_labels += [1]
    # for each writer(w), select one genuine as reference, and select one forged as negative test image
    for k in range(0, len(img_idx_each_writer_gen)):
        for m in range(0, len(img_idx_each_writer_fgd)):
            dissimilar_pairs += [[img_idx_each_writer_gen[k], img_idx_each_writer_fgd[m]]]
            dissimilar_labels += [0]
    # combine "pairs and labels of a writer" to the list
    BHSig_H.valid_pair_indices += similar_pairs
    BHSig_H.valid_pair_labels += similar_labels
    BHSig_H.valid_pair_indices += dissimilar_pairs
    BHSig_H.valid_pair_labels += dissimilar_labels

# Convert list to array
BHSig_H.valid_pair_indices = np.array(BHSig_H.valid_pair_indices)
BHSig_H.valid_pair_labels = np.array(BHSig_H.valid_pair_labels)
print('(BHSig_H) Number of valid pairs:', len(BHSig_H.valid_pair_indices))
print('(BHSig_H) Number of valid labels:', len(BHSig_H.valid_pair_labels))

# -- Generate test pairs (similar & dissimilar pairs) and corresponding pair labels (y=1 & y=0)
BHSig_H.test_pair_indices = []
BHSig_H.test_pair_labels = []
# for each writer(w), generate test pairs of #ref=1
for w in BHSig_H.test_writer_set:
    similar_pairs = []
    similar_labels = []
    dissimilar_pairs = []
    dissimilar_labels = []

    img_idx_each_writer_gen = BHSig_H.test_writer_indices_gen[w]
    img_idx_each_writer_fgd = BHSig_H.test_writer_indices_fgd[w]
    # for each writer(w), select one genuine as reference, another one genuine as positive test image
    for i in range(0, len(img_idx_each_writer_gen) - 1):
        for j in range(i+1, len(img_idx_each_writer_gen)):
            similar_pairs += [[img_idx_each_writer_gen[i], img_idx_each_writer_gen[j]]]
            similar_labels += [1]
    # for each writer(w), select one genuine as reference, and select one forged as negative test image
    for k in range(0, len(img_idx_each_writer_gen)):
        for m in range(0, len(img_idx_each_writer_fgd)):
            dissimilar_pairs += [[img_idx_each_writer_gen[k], img_idx_each_writer_fgd[m]]]
            dissimilar_labels += [0]
    # But we follow other papers to random select 276 dissimilar pairs without replacement
    random.seed(args.data_seed)
    dissimilar_pairs = random.sample(dissimilar_pairs, k=276)
    dissimilar_labels = dissimilar_labels[0:276]
    # combine "pairs and labels of a writer" to the list
    BHSig_H.test_pair_indices += similar_pairs
    BHSig_H.test_pair_labels += similar_labels
    BHSig_H.test_pair_indices += dissimilar_pairs
    BHSig_H.test_pair_labels += dissimilar_labels

# Convert list to array
BHSig_H.test_pair_indices = np.array(BHSig_H.test_pair_indices)
BHSig_H.test_pair_labels = np.array(BHSig_H.test_pair_labels)
print('(BHSig_H) Number of test pairs:', len(BHSig_H.test_pair_indices))
print('(BHSig_H) Number of test labels:', len(BHSig_H.test_pair_labels))
