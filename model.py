import torch
import torch.nn as nn
import torch.nn.functional as F



class Model_2(nn.Module):
    def __init__(self):
        super().__init__()

        # for "the shared part"
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 96, kernel_size=11, stride=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(96),
            nn.MaxPool2d(3, stride=2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size=5, padding=2, padding_mode='zeros'),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(3, stride=2)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=3, padding=1, padding_mode='zeros'),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(384)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=3, padding=1, padding_mode='zeros'),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(384)
        )

        self.conv2f = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256)
        )

        self.conv3f = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256)
        )

        # for "the global branch"
        self.conv51 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, padding=1, padding_mode='zeros'),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(3, stride=2)
        )

        # for "the regional branch"
        self.conv52 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, padding=1, padding_mode='zeros'),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(3, stride=2)
        )

        # Global Average Pooling (GAP) -> size: channel_num x 1 x 1
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # Keep for channel-wise attention
        self.ca11 = nn.Sequential(
            nn.Conv2d(256,  32, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )

        self.ca12 = nn.Sequential(
            nn.Conv2d(32,  256, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )

        # Fully connected layer
        self.fc = nn.Linear(256, output_dim)

    # Define how data flows through the layers
    def forward(self, x):
        # -- Shared part
        out1 = self.conv1(x)
        out2 = self.conv2(out1)
        out3 = self.conv3(out2)
        out4 = self.conv4(out3)
        # -- The start of global branch
        out51 = self.conv51(out4)  # dim: batch_size x 256(c) x 16(h) x 25(w)
        # -- The start of regional branch
        out52 = self.conv52(out4)  # dim: batch_size x 256(c) x 16(h) x 25(w)

        # -- Fusion of shallow features and deep features
        # Using multiplication to fuse features
        out2f = self.conv2f(out2)               # Transform to the same size as out51 & out52
        out3f = self.conv3f(out3)               # Transform to the same size as out51 & out52

        fus51 = out2f * out3f * out51
        fus52 = out2f * out3f * out52

        # -- Global Regional Channel Attention (GRCA)
        out51a = self.gap(fus51)      # dim: batch_size x 256(c) x 1(h) x 1(w)
        out51b = self.ca11(out51a)   
        out52a = self.gap(fus52)      # dim: batch_size x 256(c) x 1(h) x 1(w)
        out52b = self.ca11(out52a)    

        mul = out51b * out52b         # element-wise multiplicative fusion for latent representation (the same dim as out51b & out52b)
                                        
        out51c = self.ca12(mul) * fus51  
        out52c = self.ca12(mul) * fus52  

        # -- Global branch continues
        g = self.gap(out51c)
        g = g.view(g.size(0), -1)        
        g = F.normalize(g, p=2, dim=1)
        out_g = self.fc(g)

        # -- Regional branch continues
        # To divide region of kernel_size=(high, width) & stride (stride can adjust for overlap)

        # Divide horizontal regions
        unfold_h = torch.nn.Unfold(kernel_size=(16, 13), stride=6)
        rh = unfold_h(out52c)             
        rh = rh.view(-1, 256, 16, 13, 3)
        rh = rh.permute(0, 4, 1, 2, 3)    

        # Divide vertical regions
        unfold_v = torch.nn.Unfold(kernel_size=(8, 25), stride=4)
        rv = unfold_v(out52c)
        rv = rv.view(-1, 256, 8, 25, 3)
        rv = rv.permute(0, 4, 1, 2, 3)
        # Horizontal regions: the 1st region, the 2nd region, and the 3rd region
        r1 = rh[:, 0, :, :, :]
        r2 = rh[:, 1, :, :, :]
        r3 = rh[:, 2, :, :, :]

        r1 = self.gap(r1)
        r1 = r1.view(r1.size(0), -1)
        r1 = F.normalize(r1, p=2, dim=1)
        out_r1 = self.fc(r1)

        r2 = self.gap(r2)
        r2 = r2.view(r2.size(0), -1)
        r2 = F.normalize(r2, p=2, dim=1)
        out_r2 = self.fc(r2)

        r3 = self.gap(r3)
        r3 = r3.view(r3.size(0), -1)
        r3 = F.normalize(r3, p=2, dim=1)
        out_r3 = self.fc(r3)

        # Vertical regions: the 4th region, the 5th region, and the 6th region
        r4 = rv[:, 0, :, :, :]
        r5 = rv[:, 1, :, :, :]
        r6 = rv[:, 2, :, :, :]

        r4 = self.gap(r4)
        r4 = r4.view(r4.size(0), -1)
        r4 = F.normalize(r4, p=2, dim=1)
        out_r4 = self.fc(r4)

        r5 = self.gap(r5)
        r5 = r5.view(r5.size(0), -1)
        r5 = F.normalize(r5, p=2, dim=1)
        out_r5 = self.fc(r5)

        r6 = self.gap(r6)
        r6 = r6.view(r6.size(0), -1)
        r6 = F.normalize(r6, p=2, dim=1)
        out_r6 = self.fc(r6)

        embedding = torch.cat([out_g, out_r1, out_r2, out_r3, out_r4, out_r5, out_r6], dim=1)

        # "One input image" will generate "the following outputs"
        return embedding, out_g, out_r1, out_r2, out_r3, out_r4, out_r5, out_r6
