import torch
from torch.nn.modules import loss
from torchmetrics import ROC


class TotalLoss(loss._Loss):
    def __init__(self, weight, epsilon):
        super(TotalLoss, self).__init__()
        self.weight = weight
        self.eps = epsilon

    # -- Triplet loss: Choose positive (from 5) > hard negative - ɛ, and hard negative (the min dist from 5)
    # Using Squared L2-norm for soft-margin triplet
    def TupletLoss(self, anc, p1, p2, p3, p4, p5, n1, n2, n3, n4, n5):
        # euc_ap1: Squared Euclidean distances of anchor & positive_1 for all anchor images in a batch
        euc_ap1 = (anc - p1).pow(2).sum(1)
        euc_ap1 = torch.unsqueeze(euc_ap1, 0)
        euc_ap2 = (anc - p2).pow(2).sum(1)
        euc_ap2 = torch.unsqueeze(euc_ap2, 0)
        euc_ap3 = (anc - p3).pow(2).sum(1)
        euc_ap3 = torch.unsqueeze(euc_ap3, 0)
        euc_ap4 = (anc - p4).pow(2).sum(1)
        euc_ap4 = torch.unsqueeze(euc_ap4, 0)
        euc_ap5 = (anc - p5).pow(2).sum(1)
        euc_ap5 = torch.unsqueeze(euc_ap5, 0)
        euc_ap = torch.cat((euc_ap1, euc_ap2, euc_ap3, euc_ap4, euc_ap5), dim=0)
        # Find the maximum anc_pos distance for every column (for every anchor image)
        max_ap = torch.max(euc_ap, dim=0).values

        # euc_an1: Squared Euclidean distances of anchor & negative_1 for all anchor images in a batch
        euc_an1 = (anc - n1).pow(2).sum(1)
        euc_an1 = torch.unsqueeze(euc_an1, 0)
        euc_an2 = (anc - n2).pow(2).sum(1)
        euc_an2 = torch.unsqueeze(euc_an2, 0)
        euc_an3 = (anc - n3).pow(2).sum(1)
        euc_an3 = torch.unsqueeze(euc_an3, 0)
        euc_an4 = (anc - n4).pow(2).sum(1)
        euc_an4 = torch.unsqueeze(euc_an4, 0)
        euc_an5 = (anc - n5).pow(2).sum(1)
        euc_an5 = torch.unsqueeze(euc_an5, 0)
        euc_an = torch.cat((euc_an1, euc_an2, euc_an3, euc_an4, euc_an5), dim=0)
        # Find the minimum anc_neg distance for every column (for every anchor image)
        min_an = torch.min(euc_an, dim=0).values

        # Using "the minimum anc_neg distance" as "hard_neg".
        # If anc_pos > hard_neg - ɛ, calculate its loss. Otherwise, set as zero (no operations on these zero values)
        min_an_exp = min_an.expand(euc_ap.size()[0], -1)
        zero = torch.zeros(euc_ap.size(), device='cuda:0')
        disDiff1 = torch.where(euc_ap >= min_an_exp - self.eps, euc_ap - min_an_exp, zero)
        # Using "the maximum anc_pos distance" as "hard_pos".
        # If anc_neg < hard_pos + ɛ, calculate its loss. Otherwise, set as zero (no operations on these zero values)
        max_ap_exp = max_ap.expand(euc_an.size()[0], -1)
        disDiff2 = torch.where(euc_an <= max_ap_exp + self.eps, max_ap_exp - euc_an, zero)

        # Using soft-margin formulation, instead of the hinge function
        exp_disDiff1 = torch.where(disDiff1 != 0, torch.exp(disDiff1), zero)
        exp_disDiff2 = torch.where(disDiff2 != 0, torch.exp(disDiff2), zero)
        # sum of exp(disDiff) for each anchor/each column
        sum_exp1 = torch.sum(exp_disDiff1, dim=0)
        sum_exp2 = torch.sum(exp_disDiff2, dim=0)

        loss = torch.log(1 + sum_exp1 + sum_exp2).mean()

        return loss

    def forward(self, anc, pos1, pos2, pos3, pos4, pos5, neg1, neg2, neg3, neg4, neg5):
        # -- Global tuplet loss
        # The [1] of every anc/pos/neg image output is the global embedding
        tuplet_g = self.TupletLoss(anc[1], pos1[1], pos2[1], pos3[1], pos4[1], pos5[1],
                                   neg1[1], neg2[1], neg3[1], neg4[1], neg5[1])

        # -- Regional triplet loss
        # The [2]~[7] of every anc/pos/neg image outputs are the regional embeddings
        tuplet_r1 = self.TupletLoss(anc[2], pos1[2], pos2[2], pos3[2], pos4[2], pos5[2],
                                    neg1[2], neg2[2], neg3[2], neg4[2], neg5[2])
        tuplet_r2 = self.TupletLoss(anc[3], pos1[3], pos2[3], pos3[3], pos4[3], pos5[3],
                                    neg1[3], neg2[3], neg3[3], neg4[3], neg5[3])
        tuplet_r3 = self.TupletLoss(anc[4], pos1[4], pos2[4], pos3[4], pos4[4], pos5[4],
                                    neg1[4], neg2[4], neg3[4], neg4[4], neg5[4])
        tuplet_r4 = self.TupletLoss(anc[5], pos1[5], pos2[5], pos3[5], pos4[5], pos5[5],
                                    neg1[5], neg2[5], neg3[5], neg4[5], neg5[5])
        tuplet_r5 = self.TupletLoss(anc[6], pos1[6], pos2[6], pos3[6], pos4[6], pos5[6],
                                    neg1[6], neg2[6], neg3[6], neg4[6], neg5[6])
        tuplet_r6 = self.TupletLoss(anc[7], pos1[7], pos2[7], pos3[7], pos4[7], pos5[7],
                                    neg1[7], neg2[7], neg3[7], neg4[7], neg5[7])

        # -- Total loss
        tuplet_r = [tuplet_r1, tuplet_r2, tuplet_r3, tuplet_r4, tuplet_r5, tuplet_r6]
        loss_sum = tuplet_g + self.weight * sum(tuplet_r)

        return loss_sum


class ValidLoss(loss._Loss):
    def __init__(self):
        super(ValidLoss, self).__init__()

    def EER(self, pair_distance, label):
        # Use torchmetrics ROC (pos_label=0 is because when distance > threshold --> y_pred=0, which is judged as dissimilar)
        if 1 in label and 0 in label:
            roc = ROC(pos_label=0)
            fpr, tpr, thresholds = roc(pair_distance, label)
            # User far and frr to calculate eer => eer is when far=frr (the minimum difference of them)
            far = fpr
            # Create torch.ones in GPU
            frr = torch.ones(tpr.size(dim=0), device='cuda:0') - tpr
            diff = torch.abs(frr - far)
            min_idx = torch.argmin(diff)
            eer = (far[min_idx] + frr[min_idx]) / 2
        else:
            eer = torch.tensor(100, device='cuda:0')
        
        return eer

    def forward(self, ref, ques, y):
        # Compute the pairwise distance between two vectors using the Squared L2-norm (Squared Euclidean)
        # Model output[0] is the final embedding
        pair_dis = (ref[0] - ques[0]).pow(2).sum(1)  #.pow(0.5)
        # Calculate EER for validation loss
        error_rate = self.EER(pair_dis, y)
        return error_rate
