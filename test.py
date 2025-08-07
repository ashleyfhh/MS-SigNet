import numpy as np
import pandas as pd
from model import Model_2
from metrics import eval_metrics
from dataloader import get_data_loader
from data.preparation import CEDAR, HanSig, BHSig_B, BHSig_H
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as img
import torch
from config import arg_conf


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print(device)
args = arg_conf()
# Set Model
model = Model_2()
# Transfer model to GPU
model = model.to(device)

def err_anal(distance, threshold, y):
    # Error analysis: Check which one is wrong prediction
    sim_label = (y == 1)
    dis_label = (y == 0)

    # false positive (false accept)
    fp = (distance <= threshold) & (dis_label)
    # false negative (false reject)
    fn = (distance > threshold) & (sim_label)

    # Find the fp test pairs, including reference and questioned
    fp_ref_idx = CEDAR.test_pair_indices[fp][:, 0]            #[:, 0] is the index of reference image
    fp_ref_file = np.array(CEDAR.files_genuine)[fp_ref_idx]
    fp_ques_idx = CEDAR.test_pair_indices[fp][:, 1]           #[:, 1] is the questioned image
    fp_ques_file = np.array(CEDAR.files_forged)[fp_ques_idx]  #the questioned images of "fp" come from forged ones
    # Find the fn test pairs, including reference and questioned
    fn_ref_idx = CEDAR.test_pair_indices[fn][:, 0]             #[:, 0] is the reference image
    fn_ref_file = np.array(CEDAR.files_genuine)[fn_ref_idx]
    fn_ques_idx = CEDAR.test_pair_indices[fn][:, 1]            #[:, 1] is the questioned image
    fn_ques_file = np.array(CEDAR.files_genuine)[fn_ques_idx]  #the questioned images of "fn" come from genuine ones
    # Combine fp files with fn files & their real labels
    err_pred_ref = list(fp_ref_file) + list(fn_ref_file)
    err_pred_ques = list(fp_ques_file) + list(fn_ques_file)
    real_labels = list(y[fp]) + list(y[fn])

    # Save as a csv file
    df = pd.DataFrame({'Reference': err_pred_ref, 'Questioned': err_pred_ques, 'Real_label': real_labels})
    df.to_csv('error_analysis.csv', index=False)


def testing(predict_model, saved_model):
    # Load the saved model
    predict_model.load_state_dict(torch.load(saved_model))
    # Using evaluation mode to do inference in PyTorch
    predict_model.eval()
    # Testing data
    testloader = get_data_loader(used_mode='test', used_data=args.dataset)

    number_samples = 0
    distances = []
    labels = []

    with torch.no_grad():
        for step, (x1, x2, y) in enumerate(testloader):
            x1, x2, y = x1.to(device), x2.to(device), y.to(device)
            x1 = predict_model(x1)  # our model will return several values in a list, in which x1[0] is embedding
            x2 = predict_model(x2)  # our model will return several values in a list, in which x2[0] is embedding
            
            # Compute the pairwise distance between two vectors using the Squared L2-Norm (Squared Euclidean)
            # .extend: Combine every distance together, and combine its actual label (y) together
            distances.extend((x1[0] - x2[0]).pow(2).sum(1).cpu().tolist())
            labels.extend(y.cpu().tolist())

        # len(distances) = len(label) = the number of test pairs
        dis_array, y_array = np.array(distances), np.array(labels)

        # Use custom metrics
        Acc, FRR, FAR, EER, EER_TH, AUC = eval_metrics(dis_array, y_array)

    #     print('Max_Acc: %.4f %%' % (100 * Acc))
    #     print('FRR: %.4f %%' % (100 * FRR))
    #     print('FAR: %.4f %%' % (100 * FAR))
    #     print('EER: %.4f %%' % (100 * EER))
    #     print('EER_Threshold:', EER_TH)
    #     print('AUC: %.4f %%' % (100 * AUC))

    #     # -- Error Analysis: Check which one is wrong prediction
    #     err_anal(dis_array, EER_TH, y_array)

    # # -- Show training loss history
    # loss_image = img.imread('loss_hist.png')
    # plt.imshow(loss_image)
    # plt.show()

    return Acc, FRR, FAR, EER, EER_TH, AUC


# testing(model, args.saved_name + '.pt')
