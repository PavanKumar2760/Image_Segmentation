#%%
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sem_seg_dataset import SegmentationDataset
import segmentation_models_pytorch as smp 
import seaborn as sns
import matplotlib.pyplot as plt
import urllib.request

#%%




# %%
device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# %%
EPOCHS = 10
BS = 4

#%%
train_ds = SegmentationDataset(path_name= 'train')
train_dataloader = DataLoader(train_ds, batch_size= BS, shuffle = True)
val_ds = SegmentationDataset(path_name = 'val')
val_dataloader = DataLoader(val_ds, batch_size = BS, shuffle = True)

#%%
import os
os.environ['HTTP_PROXY'] = 'http://your_proxy_address'
os.environ['HTTPS_PROXY'] = 'https://your_proxy_address'

#%%
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

#%%
# FPN - feature pyramid network.
model = smp.FPN(encoder_name="se_resnext50_32x4d", encoder_weights="imagenet", classes=6, activation='sigmoid')


#%%
model = model.to(device)

#%%
optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001)

criterion = nn.CrossEntropyLoss()


# %%

#%%
train_losses = []
validation_losses = []

for e in range(EPOCHS):
    train_loss = 0
    val_loss = 0
    model.train()  # setting model in train mode
    for i, data in enumerate(train_dataloader):
        image_i, mask_i = data
        image = image_i.to(device)
        mask = mask_i.to(device)

        # setting grads to zero.

        optimizer.zero_grad()

        output = model(image.float())

        # defining the loss function 

        train_loss = criterion(output.float(), mask.long())

        # backpropagation
        train_loss.backward()


        # Updating the weights.
        optimizer.step()

        train_loss += train_loss.item()
    train_losses.append(train_loss)


    # validation 

    model.eval() # settign model in to validation mode.

    with torch.no_grad():
        for i, data in enumerate(val_dataloader):
            image_i, mask_i = data
            image = image_i.to(device)
            mask =  mask_i.to(device)

            # forwardpass

            output = model(image.float())

            optimizer.zero_grad()

            va_loss = criterion(output.float(), mask.long())
            val_loss += va_loss.item()
    validation_losses.append(val_loss)
    print(f"Epoch:{e}:Train Loss:{train_loss}, Validation_loss : {val_loss}")

#%%
    

# SAVING THE MODEL.
torch.save(model.state_dict(), "models/FPN_model_1.pth")



# sns.lineplot(x = range(len(train_losses)), y = train_losses).set('train_losses')

train_loss_numpy = np.array([tensor.detach().numpy() for tensor in train_losses])

#%%
sns.lineplot(x = range(len(train_loss_numpy)), y = train_loss_numpy)

#%%

# Vali_loss_numpy = np.array([tensor.detach().numpy() for tensor in validation_losses])

type(validation_losses)
#%%
sns.lineplot(x = range(len(validation_losses)), y = validation_losses)
train_losses



# %%


#%%
model.state_dict()
# %%
# now we will do the predictions.

