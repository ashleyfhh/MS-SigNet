import os
from data.preparation import CEDAR, HanSig, BHSig_B, BHSig_H
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
from config import arg_conf


# -- For CEDAR & HanSig
class signature_a():
    def __init__(self, dataset, mode, transform=None):
        self.dataset = dataset
        self.mode = mode
        self.transform = transform

        # -- training data and writer labels
        if self.mode == 'train':
            self.genuine = self.dataset.train_genuine
            self.forged = self.dataset.train_forged
            self.writer_set = self.dataset.train_writer_set
            self.writers = self.dataset.train_writers
            self.label_to_indices = self.dataset.train_writer_indices

        # -- valid data and writer labels
        elif self.mode == 'valid':
            # Use EER as Valid_loss
            self.genuine = self.dataset.valid_genuine
            self.forged = self.dataset.valid_forged
            self.pair_indices = self.dataset.valid_pair_indices
            self.pair_labels = self.dataset.valid_pair_labels

        # -- test pair data, and test labels (similar pair label: 1, dissimilar pair label: 0)
        else:
            self.genuine = self.dataset.test_genuine
            self.forged = self.dataset.test_forged
            self.pair_indices = self.dataset.test_pair_indices
            self.pair_labels = self.dataset.test_pair_labels

    def __len__(self):
        # Set "length" for sampling anchors
        if self.mode == 'train':
            return len(self.writers)
        else:
            # "Length" for valid set & test set is the number of pairs
            return len(self.pair_labels)

    def __getitem__(self, idx):
        # -- training mode
        if self.mode == 'train':
            # PyTorch will decide "idx", the index (or position) of the image in all genuine data
            # anchor_img is "the selected image", anchor_label is the "writer label" of this image
            anchor_file = self.genuine[idx]
            anchor_label = self.writers[idx].item()
            # Positive's writer label is the same as anchor's writer label, but excluding the anchor ifself
            positive_idx = np.random.choice(self.label_to_indices[anchor_label], 5, replace=False)
            while idx in positive_idx:
                positive_idx = np.random.choice(self.label_to_indices[anchor_label], 5, replace=False)
            positive_file = np.array(self.genuine)[positive_idx]
            # Negative's writer label is the same as anchor's writer label
            negative_idx = np.random.choice(self.label_to_indices[anchor_label], 5, replace=False)
            negative_file = np.array(self.forged)[negative_idx]
            # Get image files as PIL images & Transform into tensors
            positive_imgs = []
            negative_imgs = []
            # - one anchor image
            anchor_img = Image.open(os.path.join(self.dataset.path_genuine, anchor_file))
            anchor_img = self.transform(anchor_img)
            # - five positive images
            for p in positive_file:
                positive_img = Image.open(os.path.join(self.dataset.path_genuine, p))
                positive_img = self.transform(positive_img)
                positive_imgs.append(positive_img)
            # - five negative images
            for n in negative_file:
                negative_img = Image.open(os.path.join(self.dataset.path_forged, n))
                negative_img = self.transform(negative_img)
                negative_imgs.append(negative_img)

            pos_1, pos_2, pos_3, pos_4, pos_5 = positive_imgs[0], positive_imgs[1], positive_imgs[2], positive_imgs[3], positive_imgs[4]
            neg_1, neg_2, neg_3, neg_4, neg_5 = negative_imgs[0], negative_imgs[1], negative_imgs[2], negative_imgs[3], negative_imgs[4]

            return anchor_img, pos_1, pos_2, pos_3, pos_4, pos_5, neg_1, neg_2, neg_3, neg_4, neg_5, anchor_label

        # -- validation mode
        if self.mode == 'valid':
            # If similar pair (y==1), the questioned image comes from genuine, else from forged
            ref_idx = self.pair_indices[:, 0][idx]
            ref_file = self.genuine[ref_idx]
            y = self.pair_labels[idx]
            ques_idx = self.pair_indices[:, 1][idx]
            ques_file = self.genuine[ques_idx] if y == 1 else self.forged[ques_idx]

            # Get image files as PIL images
            ref_img = Image.open(os.path.join(self.dataset.path_genuine, ref_file))
            ques_img = Image.open(os.path.join(self.dataset.path_genuine, ques_file)) if y == 1 \
                else Image.open(os.path.join(self.dataset.path_forged, ques_file))

            if self.transform:
                ref_img = self.transform(ref_img)
                ques_img = self.transform(ques_img)

            return ref_img, ques_img, y

        # -- test mode
        else:
            # If similar pair (y==1), the questioned image comes from genuine, else from forged
            ref_idx = self.pair_indices[:,0][idx]
            ref_file = self.genuine[ref_idx]
            y = self.pair_labels[idx]
            ques_idx = self.pair_indices[:,1][idx]
            ques_file = self.genuine[ques_idx] if y == 1 else self.forged[ques_idx]

            # Get image files as PIL images
            ref_img = Image.open(os.path.join(self.dataset.path_genuine, ref_file))
            ques_img = Image.open(os.path.join(self.dataset.path_genuine, ques_file)) if y == 1 \
                else Image.open(os.path.join(self.dataset.path_forged, ques_file))

            if self.transform:
                ref_img = self.transform(ref_img)
                ques_img = self.transform(ques_img)

            return ref_img, ques_img, y

# -- For BHSig260 dataset
# -- BHSig260 has subdirectories, and has different numbers between genuine & forged
class signature_b():
    def __init__(self, dataset, mode, transform=None):
        self.dataset = dataset
        self.mode = mode
        self.transform = transform

        # -- training data and writer labels
        if self.mode == 'train':
            self.genuine = self.dataset.train_genuine
            self.forged = self.dataset.train_forged
            self.writer_set = self.dataset.train_writer_set
            self.writers_gen = self.dataset.train_writers_gen
            self.label_to_indices_gen = self.dataset.train_writer_indices_gen
            self.writers_fgd = self.dataset.train_writers_fgd
            self.label_to_indices_fgd = self.dataset.train_writer_indices_fgd

        # -- valid data and writer labels
        elif self.mode == 'valid':
            # Use EER as Valid_loss
            self.genuine = self.dataset.valid_genuine
            self.forged = self.dataset.valid_forged
            self.pair_indices = self.dataset.valid_pair_indices
            self.pair_labels = self.dataset.valid_pair_labels

        # -- test pair data, and test labels (similar pair label: 1, dissimilar pair label: 0)
        else:
            self.genuine = self.dataset.test_genuine
            self.forged = self.dataset.test_forged
            self.pair_indices = self.dataset.test_pair_indices
            self.pair_labels = self.dataset.test_pair_labels

    def __len__(self):
        # Set "length" for sampling anchors
        if self.mode == 'train':
            return len(self.writers_gen)
        else:
            # "Length" for valid set & test set is the number of pairs
            return len(self.pair_labels)

    def __getitem__(self, idx):
        # -- training mode
        if self.mode == 'train':
            # PyTorch will decide "idx", the index (or position) of the image in all genuine data
            # anchor_img is "the selected image", anchor_label is the "writer label" of this image
            anchor_file = self.genuine[idx]
            anchor_label = self.writers_gen[idx].item()
            positive_idx = np.random.choice(self.label_to_indices_gen[anchor_label], 5, replace=False)
            while idx in positive_idx:
                positive_idx = np.random.choice(self.label_to_indices_gen[anchor_label], 5, replace=False)
            positive_file = np.array(self.genuine)[positive_idx]
            negative_idx = np.random.choice(self.label_to_indices_fgd[anchor_label], 5, replace=False)
            negative_file = np.array(self.forged)[negative_idx]
            # Get image files as PIL images & Transform into tensors
            positive_imgs = []
            negative_imgs = []
            # - one anchor image
            anchor_img = Image.open(os.path.join(self.dataset.path_genuine, anchor_file.split('_')[1].zfill(3), anchor_file))
            anchor_img = self.transform(anchor_img)
            # - five positive images
            for p in positive_file:
                positive_img = Image.open(os.path.join(self.dataset.path_genuine, p.split('_')[1].zfill(3), p))
                positive_img = self.transform(positive_img)
                positive_imgs.append(positive_img)
            # - five negative images
            for n in negative_file:
                negative_img = Image.open(os.path.join(self.dataset.path_forged, n.split('_')[1].zfill(3), n))
                negative_img = self.transform(negative_img)
                negative_imgs.append(negative_img)

            pos_1, pos_2, pos_3, pos_4, pos_5 = positive_imgs[0], positive_imgs[1], positive_imgs[2], positive_imgs[3], positive_imgs[4]
            neg_1, neg_2, neg_3, neg_4, neg_5 = negative_imgs[0], negative_imgs[1], negative_imgs[2], negative_imgs[3], negative_imgs[4]

            return anchor_img, pos_1, pos_2, pos_3, pos_4, pos_5, neg_1, neg_2, neg_3, neg_4, neg_5, anchor_label

        # -- validation mode
        if self.mode == 'valid':
            # If similar pair (y==1), the questioned image comes from genuine, else from forged
            ref_idx = self.pair_indices[:, 0][idx]
            ref_file = self.genuine[ref_idx]
            y = self.pair_labels[idx]
            ques_idx = self.pair_indices[:, 1][idx]
            ques_file = self.genuine[ques_idx] if y == 1 else self.forged[ques_idx]

            # Get image files as PIL images
            ref_img = Image.open(os.path.join(self.dataset.path_genuine, ref_file.split('_')[1].zfill(3), ref_file))
            ques_img = Image.open(os.path.join(self.dataset.path_genuine, ques_file.split('_')[1].zfill(3), ques_file)) if y == 1 \
                else Image.open(os.path.join(self.dataset.path_forged, ques_file.split('_')[1].zfill(3), ques_file))

            if self.transform:
                ref_img = self.transform(ref_img)
                ques_img = self.transform(ques_img)

            return ref_img, ques_img, y

        # -- test mode
        else:
            # If similar pair (y==1), the questioned image comes from genuine, else from forged
            ref_idx = self.pair_indices[:,0][idx]
            ref_file = self.genuine[ref_idx]
            y = self.pair_labels[idx]
            ques_idx = self.pair_indices[:,1][idx]
            ques_file = self.genuine[ques_idx] if y == 1 else self.forged[ques_idx]

            # Get image files as PIL images
            ref_img = Image.open(os.path.join(self.dataset.path_genuine, ref_file.split('_')[1].zfill(3), ref_file))
            ques_img = Image.open(os.path.join(self.dataset.path_genuine, ques_file.split('_')[1].zfill(3), ques_file)) if y == 1 \
                else Image.open(os.path.join(self.dataset.path_forged, ques_file.split('_')[1].zfill(3), ques_file))

            if self.transform:
                ref_img = self.transform(ref_img)
                ques_img = self.transform(ques_img)

            return ref_img, ques_img, y


# -- Load data & transform
def get_data_loader(used_mode, used_data):
    args = arg_conf()
    np.random.seed(args.random_seed)

    preprocess = transforms.Compose(
        [transforms.Grayscale(num_output_channels=1),
        transforms.Resize((150, 220)),
        transforms.ToTensor()]
        )

    if used_data == 'hansig':
        data = signature_a(HanSig, mode=used_mode, transform=preprocess)
    elif used_data == 'cedar':
        data = signature_a(CEDAR, mode=used_mode, transform=preprocess)
    elif used_data == 'bengali':
        data = signature_b(BHSig_B, mode=used_mode, transform=preprocess)
    elif used_data == 'hindi':
        data = signature_b(BHSig_H, mode=used_mode, transform=preprocess)
    else:
        raise ValueError(f'Unknow dataset {used_data}')

    if used_mode == 'train':
        batch_size = args.batch_train
        is_shuffle = True
    elif used_mode == 'valid':
        batch_size = args.batch_valid
        is_shuffle = True  
    elif used_mode == 'test':
        batch_size = args.batch_test
        is_shuffle = False   

    loader = DataLoader(data, pin_memory=True, batch_size=batch_size, shuffle=is_shuffle)

    return loader
