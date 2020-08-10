#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import datetime


# In[2]:


train_transformer = transforms.Compose([transforms.RandomRotation(45),
                                                    transforms.RandomHorizontalFlip(p = 0.4),
                                                    transforms.ColorJitter(brightness = 0.5, contrast = 0.5, hue = 0.3, saturation = 1.5),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5])])

test_transformer = transforms.Compose([transforms.ToTensor(),
                                       transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5])])


# In[3]:


# Download training data
train_data = torchvision.datasets.CIFAR10(root = '.',
                                          download = True,
                                          train = True,
                                          transform = transforms.ToTensor()   
                                         )


# In[4]:


test_data = torchvision.datasets.CIFAR10(root = '.',
                                         download = True,
                                         train = False,
                                         transform = transforms.ToTensor()
                                        )


# In[5]:


train_data.data[0]


# In[6]:


train_data.data.shape


# In[7]:


# Number of classes
K = len(set(train_data.targets))
K


# In[8]:


test_data.data.shape


# In[9]:


# Data Loader

batch_size = 128

train_loader = torch.utils.data.DataLoader(dataset = train_data,
                                           batch_size = batch_size,
                                           shuffle = True
                                          )

test_loader = torch.utils.data.DataLoader(dataset = test_data,
                                          batch_size = batch_size,
                                          shuffle = False
                                         )


# In[10]:


for inputs, targets in train_loader:
    break


# In[11]:


inputs.shape


# ### Define the CNN model
# 
# This model used as a presentation doesn't have complex architecture because I tried stacking more layers but it takes like forever to finish training on my laptop...

# In[12]:


class CNN(nn.Module):
    def __init__(self, K):
        super(CNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size = 3, stride = 2),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, stride = 2),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, stride =2),
            nn.ReLU()
        )
        
        self.dense_layers = nn.Sequential(
            nn.Dropout(p = 0.2),
            nn.Linear(in_features = 128*3*3, out_features = 1024),
            nn.ReLU(),
            nn.Dropout(p = 0.2),
            nn.Linear(in_features = 1024, out_features = K)
        )
        
    def forward(self, X):
        out = self.conv_layers(X)
        out = out.view(out.shape[0], -1)
        out = self.dense_layers(out)
        return out


# In[13]:


# Instantiate model
model = CNN(K)


# In[14]:


# loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())


# In[15]:


# Define a loop function 
def batch_gd(train_loader, test_loader, criterion, optimizer, epochs):
    train_losses = np.zeros(epochs)
    test_losses = np.zeros(epochs)
    
    # Traning loop
    for i in range(epochs):
        train_loss = []
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            train_loss.append(loss.item())
            
            loss.backward()
            optimizer.step()
            
        train_loss = np.mean(train_loss)
        train_losses[i] = train_loss
      
        test_loss = []
        for inputs, targets in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss.append(loss.item())
            
        test_loss = np.mean(test_loss)
        test_losses[i] = test_loss
        
        print(f'Epoch: {i+1}/{epochs}, train_loss: {train_loss: .3f}, test_loss:{test_loss: .3f}')
    return train_losses, test_losses


# In[16]:


epochs = 8

train_losses, test_losses = batch_gd(train_loader, test_loader, criterion, optimizer, epochs)


# In[17]:


plt.plot(train_losses, label = 'Train Loss')
plt.plot(test_losses, label = 'Test Loss')
plt.legend()
plt.show()


# In[18]:


# Accuracy

n_correct = 0
n_total = 0

for inputs, targets in train_loader:
    outputs = model(inputs)
    _, predictions = torch.max(outputs, 1)
    correct = (predictions == targets).sum()
    n_correct += correct.item()
    
    n_total += len(targets)
    
train_accuracy = n_correct/n_total
print('train_accuracy: %s' % (train_accuracy))

for inputs, targets in test_loader:
    outputs = model(inputs)
    _, predictions = torch.max(outputs, 1)
    correct = (predictions == targets).sum()
    n_correct += correct.item()
    n_total += len(targets)
    
test_accuracy = n_correct/n_total

print('test_accuracy: %s' % (test_accuracy))


# In[19]:


from sklearn.metrics import confusion_matrix
import itertools

x_test = test_data.data
y_test = test_data.targets
p_test = np.array([])

for inputs, targets in test_loader:
    outputs = model(inputs)
    _, predictions = torch.max(outputs, 1)
    p_test = np.concatenate((p_test, predictions.numpy()))


# In[20]:


# Define a function to plot confusion matrix
def plot_confusion_matrix(cm, classes, normalize = False, cmap = plt.cm.Blues, title = 'Confusion matirx'):
    plt.imshow(cm, interpolation='nearest', cmap = cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation = 45)
    plt.yticks(tick_marks, classes)
    
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max()/2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(i, j, format(cm[i, j], fmt), color = 'orange' if cm[i, j] > thresh else 'black',
                 horizontalalignment = 'center')
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Prediction label')
    plt.show()


# In[21]:


cm = confusion_matrix(y_test, p_test)
plot_confusion_matrix(cm, list(range(10)))


# In[22]:


# label mapping
labels = '''airplane
    automobile
    bird
    cat
    deer
    dog
    frog
    horse
    ship
    truck'''.split()


# In[23]:


# show some misclassified examples
p_test = p_test.astype(np.uint8)
misclassified_id = np.where(p_test != y_test)[0]
im_id = np.random.choice(misclassified_id)
print(im_id)
plt.imshow(x_test[im_id].reshape(32, 32, 3))
print('actual label: %s' % (labels[y_test[im_id]]))
print('predicted label: %s' % (labels[p_test[im_id]]))


# In[24]:


from torchsummary import summary
summary(model, (3, 32, 32))


# In[ ]:





# In[ ]:




