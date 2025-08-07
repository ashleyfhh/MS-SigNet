import numpy as np
from model import Model_2
from tqdm import tqdm
from dataloader import get_data_loader
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from co_tuplet_loss import TotalLoss, ValidLoss
from config import arg_conf


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print(device)
args = arg_conf()

# Set initial weights
def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
    # if isinstance(m, nn.Conv2d):
    #     nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
    #     nn.init.constant_(m.bias, 0.)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(m.bias, 0.)
    elif isinstance(m, nn.ReLU):
        pass


# Set Model
model = Model_2()
print(model)
# Set initial weights for Conv layers
model.apply(init_weights)
# Transfer model to GPU
model = model.to(device)

#summary(model, (1, 150, 220))

optimizer = optim.Adam(model.parameters(), lr=args.lr)
# Appy learning decay (lr_new = lr_old * gamma) at the specific epochs/the milestone
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15, 30], gamma=args.lr_decay)
criterion_1 = TotalLoss(args.loss_weight, args.loss_epsilon)
criterion_2 = ValidLoss()

# Plot loss in the training process
def loss_plot(loss_list1, loss_list2):
    plt.figure()
    plt.title("Training and validation loss")
    plt.plot(list(range(1, len(loss_list1)+1)), loss_list1, label='train')
    plt.plot(list(range(1, len(loss_list2)+1)), loss_list2, label='valid')
    plt.xlabel('epoch', fontsize=12)
    plt.ylabel('loss', fontsize=12)
    plt.legend(loc='best')
    plt.savefig('loss_hist.png')

# Train the model
def training():
    num_epochs = args.n_epoch
    patience = args.patience
    valid_loss_min = float('inf')
    previous_loss = args.prev_loss
    trigger_times = args.trig_times

    trainloader = get_data_loader(used_mode='train', used_data=args.dataset)
    validloader = get_data_loader(used_mode='valid', used_data=args.dataset)

    train_loss = []
    valid_loss = []

    for epoch in range(1, num_epochs + 1):
        # Training: Using train mode of PyTorch
        model.train()
        train_step_loss = []
        # num_steps in one epoch = num_samples / batch_size
        for step, (anc, p1, p2, p3, p4, p5, n1, n2, n3, n4, n5, anchor_label) in enumerate(tqdm(trainloader, desc=f'Epoch {epoch}', leave=False)):
            # Transfer tensor from CPU to GPU
            anc = anc.to(device)
            p1 = p1.to(device)
            p2 = p2.to(device)
            p3 = p3.to(device)
            p4 = p4.to(device)
            p5 = p5.to(device)
            n1 = n1.to(device)
            n2 = n2.to(device)
            n3 = n3.to(device)
            n4 = n4.to(device)
            n5 = n5.to(device)

            # Clear the gradients
            optimizer.zero_grad()
            # Forward Pass
            anc_out = model(anc)     # return a list of the global embedding & several regional embeddings
            p1_out = model(p1)         
            p2_out = model(p2)         
            p3_out = model(p3)
            p4_out = model(p4)
            p5_out = model(p5)
            n1_out = model(n1)
            n2_out = model(n2)
            n3_out = model(n3)
            n4_out = model(n4)
            n5_out = model(n5)

            # Calculate loss in a step
            loss = criterion_1(anc_out, p1_out, p2_out, p3_out, p4_out, p5_out,
                               n1_out, n2_out, n3_out, n4_out, n5_out)

            # Back pass to calculate gradients
            loss.backward()
            # Update Weights
            optimizer.step()

            # Append "loss in a step" into list
            train_step_loss.append(loss.detach().cpu().numpy())

        # "loss of one epoch" is the average of all "loss in a step"
        current_train_loss = np.mean(train_step_loss)
        train_loss.append(current_train_loss)
        print("Epoch: {}/{} - Train_Loss: {:.4f}".format(epoch, num_epochs, current_train_loss))

        torch.save(model.state_dict(), args.saved_name + "{:.0f}".format(epoch) + '.pt')

    #     # Validation: Using evaluation mode of PyTorch
    #     model.eval()
    #     valid_step_loss = []
    #     with torch.no_grad():
    #         # x1: reference image, x2: questioned image, y: pair label (1 is similar & 0 is dissimilar)
    #         for step, (x1, x2, y) in enumerate(validloader):
    #             x1, x2, y = x1.to(device), x2.to(device), y.to(device)
        
    #             # Validation steps are also similar to training steps
    #             # But just make a forward pass and calculate loss
    #             x1 = model(x1)  # our model will return several values in a list, in which x1[0] is embedding
    #             x2 = model(x2)  # our model will return several values in a list, in which x2[0] is embedding
    #             # Calculate loss in a step (x1[0] & x2[0] has been written in loss function)
    #             loss_v = criterion_2(x1, x2, y)
    #             # Append "loss in a step" into list
    #             valid_step_loss.append(loss_v.detach().cpu().numpy())
        
    #         # "loss of one epoch" is the average of all "loss in a step"
    #         valid_step_loss = np.array(valid_step_loss).astype('float')
    #         valid_step_loss[valid_step_loss == 100] = np.nan
    #         # Calculate mean value ignoring np.nan
    #         current_valid_loss = np.nanmean(valid_step_loss)
    #         valid_loss.append(current_valid_loss)
    #         print("------------- Valid_Loss: {:.4f}\n".format(current_valid_loss))
        
    #         # # Checkpoint: Save the model of the min valid_loss
    #         # if current_valid_loss < valid_loss_min:
    #         #     valid_loss_min = current_valid_loss
    #         #     torch.save(model.state_dict(), args.saved_name + "{:.4f}".format(current_valid_loss) + '.pt')
    #         #     print('Improvement in Valid_Loss, save model', '\n')
    #         # Early stopping
    #         if current_valid_loss > previous_loss:
    #             trigger_times += 1
    #             if trigger_times >= patience:
    #                 print('Valid_loss has no progress. Early stopping!')
    #                 break
    #         elif current_valid_loss <= previous_loss:
    #             trigger_times = 0
        
    #         previous_loss = current_valid_loss

    # # Apply learning decay at the specific epochs
    # scheduler.step()

    # # Loss plot of the training process
    # loss_plot(train_loss, valid_loss)

training()
