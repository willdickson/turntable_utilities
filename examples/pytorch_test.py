import sys
import time
import copy
import h5py
import numpy as np
import torch
import torchvision

def train_model(model, criterion, optimizer, scheduler, dataloader_dict, dataset_sizes, device, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    epoch_loss =  0.0 

    for epoch in range(num_epochs):
        #print('Epoch {}/{}'.format(epoch, num_epochs - 1))

        #print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            itercount = 0
            for inputs, labels in dataloader_dict[phase]:

                print(' epoch:  {}/{},  phase: {},  count: {}, last eloss: {:1.3f}'.format(epoch, num_epochs, phase, itercount, epoch_loss))

                inputs = inputs.to(device)
                labels = labels.float().to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                print(' ', loss.item())
                #running_loss += loss.item() * inputs.size(0)
                running_loss += loss.item() 
                # running_corrects += torch.sum(preds == labels.data)
                itercount += 1

            if phase == 'train':
                scheduler.step()

            #epoch_loss = running_loss / dataset_sizes[phase]
            epoch_loss = running_loss 
            #epoch_acc = running_corrects.double() / dataset_sizes[phase]

            #print('{} Loss: {:.4f} Acc: {:.4f}'.format(
            #    phase, epoch_loss, epoch_acc))
            print('{} Loss: {:.4f}'.format(phase, epoch_loss))

            # deep copy the model
            #if phase == 'val' and epoch_acc > best_acc:
            #    best_acc = epoch_acc
            #    best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    #print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model



# -----------------------------------------------------------------------------
if __name__ == '__main__':


    data_filename = sys.argv[1]

    h5data = h5py.File(data_filename,'r')
    frame_array = np.array(h5data['frame'])
    image_array = np.array(h5data['image'])
    probs_array = np.array(h5data['probs'])
    h5data.close()

    data_transforms = { 
            'train': torchvision.transforms.Compose([ 
                torchvision.transforms.ToPILImage(),
                #torchvision.transforms.Resize(224),
                #torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
                ]), 
            'val': torchvision.transforms.Compose([ 
                torchvision.transforms.ToPILImage(),
                #torchvision.transforms.Resize(224),
                #torchvision.transforms.CenterCrop(224),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
                ]), 
            }

    #n = image_array.shape[0]
    n = 200 
    train_image_list = [image_array[i,:,:,:] for i in range(n//2)]
    train_probs_list = [probs_array[i,:,:].flatten() for i in range(n//2)]
    train_image_tensor = torch.stack([data_transforms['train'](torch.tensor(image)) for image in train_image_list])
    train_probs_tensor = torch.stack([torch.tensor(probs) for probs in train_probs_list])
    train_dataset = torch.utils.data.TensorDataset(train_image_tensor,train_probs_tensor) # create your datset
    train_dataloader = torch.utils.data.DataLoader(train_dataset) # create your dataloader

    val_image_list = [image_array[i,:,:,:] for i in range(n//2,n)]
    val_probs_list = [probs_array[i,:,:].flatten() for i in range(n//2,n)]
    val_image_tensor = torch.stack([data_transforms['val'](torch.tensor(image)) for image in val_image_list])
    val_probs_tensor = torch.stack([torch.tensor(probs) for probs in val_probs_list])
    val_dataset = torch.utils.data.TensorDataset(val_image_tensor,val_probs_tensor) # create your datset
    val_dataloader = torch.utils.data.DataLoader(val_dataset) # create your dataloader

    dataloader_dict = {'train': train_dataloader, 'val': val_dataloader}
    dataset_sizes = {'train': len(train_image_list), 'val': len(val_image_list)}

    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")


    model_conv = torchvision.models.resnet18(pretrained=True)
    for param in model_conv.parameters():
        param.requires_grad = False

    # Parameters of newly constructed modules have requires_grad=True by default
    num_ftrs = model_conv.fc.in_features
    #model_conv.fc = torch.nn.Linear(num_ftrs, probs_array.shape[1]*probs_array.shape[2])
    model_conv.fc = torch.nn.Sequential(
            torch.nn.Linear(num_ftrs, probs_array.shape[1]*probs_array.shape[2]),
            torch.nn.Softmax(),
            )

    model_conv = model_conv.to(device)
    #criterion = torch.nn.CrossEntropyLoss()
    criterion = torch.nn.MSELoss()
    
    # Observe that only parameters of final layer are being optimized as
    # opposed to before.
    optimizer_conv = torch.optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)
    
    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

    model_conv = train_model(model_conv, criterion, optimizer_conv, exp_lr_scheduler, dataloader_dict, dataset_sizes, device, num_epochs=25)




