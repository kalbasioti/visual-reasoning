import numpy as np
import wandb
import random

seed = 1
random.seed(seed)

def VPROM_splits():
    N = 235000  # available RPMs

    # split training/testing sets ratio 2:1 for the Neutral Regime of V-PROM
    N_test = int(np.floor(N/3))
    ind_test = random.sample(range(N),N_test)

    # V-PROM split files
    counting_labels_path = 'datasets/counting_gt_labels.npy'
    objects_labels_path = 'datasets/obj_gt_labels.npy'
    attr_labels_path = 'datasets/attr_gt_labels.npy'

    labels_counting = np.load(counting_labels_path, allow_pickle=True)
    labels_counting = labels_counting.tolist()
    labels_obj = np.load(objects_labels_path, allow_pickle=True)
    labels_obj = labels_obj.tolist()
    labels_attr = np.load(attr_labels_path, allow_pickle=True)
    labels_attr = labels_attr.tolist()

    fname_imgs_test = []
    fname_targets_test = []
    fname_imgs_train = []
    fname_targets_train = []

    for label in labels_counting:
        lbls = labels_counting[label]
        for sublabel in lbls:
            l = int(sublabel)
            res = l in ind_test
            if res:
                fname_imgs_test.append(lbls[sublabel]['im_path'])
                fname_targets_test.append(lbls[sublabel]['label'])
            else:
                fname_imgs_train.append(lbls[sublabel]['im_path'])
                fname_targets_train.append(lbls[sublabel]['label'])

    for label in labels_obj:
        lbls = labels_obj[label]
        for sublabel in lbls:
            l = int(sublabel)
            res = l in ind_test
            if res:
                fname_imgs_test.append(lbls[sublabel]['im_path'])
                fname_targets_test.append(lbls[sublabel]['label'])
            else:
                fname_imgs_train.append(lbls[sublabel]['im_path'])
                fname_targets_train.append(lbls[sublabel]['label'])

    for label in labels_attr:
        rule = labels_attr[label]
        for sublabel in rule:
            lbls = rule[sublabel]
            for subsublabel in lbls:
                l = int(subsublabel)
                res = l in ind_test
                if res:
                    fname_imgs_test.append(lbls[subsublabel]['im_path'])
                    fname_targets_test.append(lbls[subsublabel]['label'])
                else:
                    fname_imgs_train.append(lbls[subsublabel]['im_path'])
                    fname_targets_train.append(lbls[subsublabel]['label'])

    return fname_imgs_test, fname_imgs_train, fname_targets_test, fname_targets_train

def initialize_wandb(args):
    if args.use_wandb:
        wandb.init(
            project="your_project_name",
            config=vars(args),
            sync_tensorboard=True,
            name=args.name,
            reinit=True,
        )
