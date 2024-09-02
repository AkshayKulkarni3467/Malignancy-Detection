"""
Contains functions for training and testing a PyTorch model.
"""
import pickle
import torch
from pathlib import Path
from tqdm.auto import tqdm
from typing import Dict, List, Tuple

def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str):
    """Saves a PyTorch model to a target directory.

    Args:
    model: A target PyTorch model to save.
    target_dir: A directory for saving the model to.
    model_name: A filename for the saved model. Should include
      either ".pth" or ".pt" as the file extension.

    Example usage:
    save_model(model=model_0,
               target_dir="models",
               model_name="05_going_modular_tingvgg_model.pth")
    """
    # Create target directory
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True,
                        exist_ok=True)

    # Create model save path
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
    model_save_path = target_dir_path / model_name

    # Save the model state_dict()
    print(f"[INFO] Saving model to: {model_save_path}")
    model.to('cpu')
    torch.save(obj=model.state_dict(),
             f=model_save_path)
    model.to('cuda')

def train_step(model: torch.nn.Module, 
               dataloader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               optimizer: torch.optim.Optimizer,
               device: torch.device) -> Tuple[float, float]:
    """Trains a PyTorch model for a single epoch.

    Turns a target PyTorch model to training mode and then
    runs through all of the required training steps (forward
    pass, loss calculation, optimizer step).

    Args:
    model: A PyTorch model to be trained.
    dataloader: A DataLoader instance for the model to be trained on.
    loss_fn: A PyTorch loss function to minimize.
    optimizer: A PyTorch optimizer to help minimize the loss function.
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    A tuple of training loss and training accuracy metrics.
    In the form (train_loss, train_accuracy). For example:

    (0.1112, 0.8743)
    """
    # Put model in train mode
    model.train()

    # Setup train loss and train accuracy values
    train_loss, train_acc = 0, 0
    precision_0,precision_1, precision_2, precision_3, precision_4,precision_5,precision_6,precision_7 = 0,0,0,0,0,0,0,0
    recall_0,recall_1,recall_2,recall_3,recall_4,recall_5,recall_6,recall_7 = 0,0,0,0,0,0,0,0
    f1_0,f1_1,f1_2,f1_3,f1_4,f1_5,f1_6,f1_7 = 0,0,0,0,0,0,0,0
    tpr = 0
    fpr = 0

    # Loop through data loader data batches
    for batch, (X, y) in enumerate(dataloader):
        # Send data to target device
        X, y = X.to(device), y.to(device)

        # 1. Forward pass
        y_pred = model(X)

        # 2. Calculate  and accumulate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item() 

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        # Calculate and accumulate accuracy metric across all batches
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item()/len(y_pred)
        tp_0,tp_1,tp_2,tp_3,tp_4,tp_5,tp_6,tp_7 = 0,0,0,0,0,0,0,0
        fn_0,fn_1,fn_2,fn_3,fn_4,fn_5,fn_6,fn_7 =0,0,0,0,0,0,0,0
        fp_0,fp_1,fp_2,fp_3,fp_4,fp_5,fp_6,fp_7 = 0,0,0,0,0,0,0,0
        tn_0,tn_1,tn_2,tn_3,tn_4,tn_5,tn_6,tn_7 = 0,0,0,0,0,0,0,0
        y_pred_class = y_pred_class.cpu().numpy()
        y = y.cpu().numpy()
        for i,j in zip(y_pred_class,y):
          if (i == j) and j == 0:
            tp_0 +=1
          if (i ==j) and j == 1:
            tp_1 += 1
          if (i == j) and j == 2:
            tp_2+=1
          if (i == j) and j == 3:
            tp_3+=1
          if (i == j) and j == 4:
            tp_4+=1
          if (i==j) and j == 5:
            tp_5+=1
          if (i == j) and j == 6:
            tp_6+=1
          if (i == j) and j == 7:
            tp_7+=1
            
        for i,j in zip(y_pred_class,y):
          if (i != j) and j == 0:
            fn_0 +=1
          if (i !=j) and j == 1:
            fn_1 += 1
          if (i != j) and j == 2:
            fn_2+=1
          if (i != j) and j == 3:
            fn_3+=1
          if (i != j) and j == 4:
            fn_4+=1
          if (i!=j) and j == 5:
            fn_5+=1
          if (i != j) and j == 6:
            fn_6+=1
          if (i != j) and j == 7:
            fn_7+=1
            
        for i,j in zip(y_pred_class,y):
          if (i != j) and i == 0:
            fp_0 +=1
          if (i !=j) and i == 1:
            fp_1 += 1
          if (i != j) and i == 2:
            fp_2+=1
          if (i != j) and i == 3:
            fp_3+=1
          if (i != j) and i == 4:
            fp_4+=1
          if (i!=j) and i == 5:
            fp_5+=1
          if (i != j) and i == 6:
            fp_6+=1
          if (i != j) and i == 7:
            fp_7+=1
            
        for i,j in zip(y_pred_class,y):
          if (i != j) and i != 0:
            tn_0 +=1
          if (i !=j) and i != 1:
            tn_1 += 1
          if (i != j) and i != 2:
            tn_2+=1
          if (i != j) and i != 3:
            tn_3+=1
          if (i != j) and i != 4:
            tn_4+=1
          if (i!=j) and i != 5:
            tn_5+=1
          if (i != j) and i != 6:
            tn_6+=1
          if (i != j) and i != 7:
            tn_7+=1
            
        recall_0 = recall_0 + (tp_0/(tp_0 + fn_0 + 1))
        recall_1 = recall_1 + (tp_1/(tp_1 + fn_1 + 1))
        recall_2 = recall_2 +(tp_2/(tp_2 + fn_2 + 1))
        recall_3 = recall_3 +(tp_3/(tp_3 + fn_3 + 1))
        recall_4 = recall_4 +(tp_4/(tp_4 + fn_4 + 1))
        recall_5 = recall_5 +(tp_5/(tp_5 + fn_5 + 1))
        recall_6 = recall_6 +(tp_6/(tp_6 + fn_6 + 1))
        recall_7 = recall_7 +(tp_7/(tp_7 + fn_7 + 1))
        
        precision_0 = precision_0 + (tp_0/(tp_0 + fp_0 + 1))
        precision_1 = precision_1 + (tp_1/(tp_1 + fp_1 + 1))
        precision_2 = precision_2 + (tp_2/(tp_2 + fp_2 + 1)) 
        precision_3 = precision_3 + (tp_3/(tp_3 + fp_3 + 1))
        precision_4 = precision_4 + (tp_4/(tp_4 + fp_4 + 1))
        precision_5 = precision_5 + (tp_5/(tp_5 + fp_5 + 1))
        precision_6 = precision_6 + (tp_6/(tp_6 + fp_6 + 1))
        precision_7 = precision_7 + (tp_7/(tp_7 + fp_7 + 1))
        
        
        f1_0 = f1_0 + ((2*recall_0*precision_0)/(recall_0+precision_0+1))
        f1_1 = f1_1 + ((2*recall_1*precision_1)/(recall_1+precision_1+1))
        f1_2 = f1_2 + ((2*recall_2*precision_2)/(recall_2+precision_2+1))
        f1_3 = f1_3 + ((2*recall_3*precision_3)/(recall_3+precision_3+1))
        f1_4 = f1_4 + ((2*recall_4*precision_4)/(recall_4+precision_4+1))
        f1_5 = f1_5 + ((2*recall_5*precision_5)/(recall_5+precision_5+1))
        f1_6 = f1_6 + ((2*recall_6*precision_6)/(recall_6+precision_6+1))
        f1_7 = f1_7 + ((2*recall_7*precision_7)/(recall_7+precision_7+1))
        
        tpr = tpr + ((tp_0 + tp_1 + tp_2 + tp_3+tp_4+tp_5+tp_6+tp_7)/(tp_0 + tp_1 + tp_2 + tp_3+tp_4+tp_5+tp_6+tp_7 + fn_0+fn_1+fn_2+fn_3+fn_4+fn_5+fn_6+fn_7+1))
        fpr = fpr + ((fp_0 + fp_1 + fp_2 + fp_3 + fp_4 + fp_5 + fp_6 + fp_7)/(fp_0 + fp_1 + fp_2 + fp_3 + fp_4 + fp_5 + fp_6 + fp_7+tn_0+tn_1+tn_2+tn_3+tn_4+tn_5+tn_6+tn_7+1))

    # Adjust metrics to get average loss and accuracy per batch 
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    recall_0 = recall_0 / len(dataloader)
    recall_1 = recall_1 / len(dataloader)
    recall_2 = recall_2 / len(dataloader)
    recall_3 = recall_3 / len(dataloader)
    recall_4 = recall_4 / len(dataloader)
    recall_5 = recall_5 / len(dataloader)
    recall_6 = recall_6 / len(dataloader)
    recall_7 = recall_7 / len(dataloader)
    
    precision_0 = precision_0 / len(dataloader)
    precision_1 = precision_1 / len(dataloader)
    precision_2 = precision_2 / len(dataloader)
    precision_3 = precision_3 / len(dataloader)
    precision_4 = precision_4 / len(dataloader)
    precision_5 = precision_5 / len(dataloader)
    precision_6 = precision_6 / len(dataloader)
    precision_7 = precision_7 / len(dataloader)
    
    f1_0 = f1_0 /len(dataloader)
    f1_1 = f1_1 /len(dataloader)
    f1_2 = f1_2 /len(dataloader)
    f1_3 = f1_3 /len(dataloader)
    f1_4 = f1_4 /len(dataloader)
    f1_5 = f1_5 /len(dataloader)
    f1_6 = f1_6 /len(dataloader)
    f1_7 = f1_7 /len(dataloader)
    
    tpr = tpr / len(dataloader)
    fpr = fpr / len(dataloader)
    
    return train_loss, train_acc,recall_0,recall_1,recall_2,recall_3,recall_4,recall_5,recall_6,recall_7,precision_0,precision_1,precision_2,precision_3,precision_4,precision_5,precision_6,precision_7,f1_0,f1_1,f1_2,f1_3,f1_4,f1_5,f1_6,f1_7,tpr,fpr

def test_step(model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader, 
              loss_fn: torch.nn.Module,
              device: torch.device) -> Tuple[float, float]:
    """Tests a PyTorch model for a single epoch.

    Turns a target PyTorch model to "eval" mode and then performs
    a forward pass on a testing dataset.

    Args:
    model: A PyTorch model to be tested.
    dataloader: A DataLoader instance for the model to be tested on.
    loss_fn: A PyTorch loss function to calculate loss on the test data.
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    A tuple of testing loss and testing accuracy metrics.
    In the form (test_loss, test_accuracy). For example:

    (0.0223, 0.8985)
    """
    # Put model in eval mode
    model.eval() 

    # Setup test loss and test accuracy values
    test_loss, test_acc = 0, 0

    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for batch, (X, y) in enumerate(dataloader):
            # Send data to target device
            X, y = X.to(device), y.to(device)

            # 1. Forward pass
            test_pred_logits = model(X)

            # 2. Calculate and accumulate loss
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()

            # Calculate and accumulate accuracy
            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))

    # Adjust metrics to get average loss and accuracy per batch 
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc

def train(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          test_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          device: torch.device) -> Dict[str, List]:
    """Trains and tests a PyTorch model.

    Passes a target PyTorch models through train_step() and test_step()
    functions for a number of epochs, training and testing the model
    in the same epoch loop.

    Calculates, prints and stores evaluation metrics throughout.

    Args:
    model: A PyTorch model to be trained and tested.
    train_dataloader: A DataLoader instance for the model to be trained on.
    test_dataloader: A DataLoader instance for the model to be tested on.
    optimizer: A PyTorch optimizer to help minimize the loss function.
    loss_fn: A PyTorch loss function to calculate loss on both datasets.
    epochs: An integer indicating how many epochs to train for.
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    A dictionary of training and testing loss as well as training and
    testing accuracy metrics. Each metric has a value in a list for 
    each epoch.
    In the form: {train_loss: [...],
              train_acc: [...],
              test_loss: [...],
              test_acc: [...]} 
    For example if training for epochs=2: 
             {train_loss: [2.0616, 1.0537],
              train_acc: [0.3945, 0.3945],
              test_loss: [1.2641, 1.5706],
              test_acc: [0.3400, 0.2973]} 
    """
    # Create empty results dictionary
    results = {"train_loss": [],
               "train_acc": [],
               'recall_0': [],
               'recall_1': [],
               'recall_2': [],
               'recall_3' : [],
               'recall_4' : [],
               'recall_5' : [],
               'recall_6' : [],
               'recall_7' : [],
               'precision_0' : [],
               'precision_1' : [],
               'precision_2' : [],
               'precision_3' : [],
               'precision_4' : [],
               'precision_5' : [],
               'precision_6' : [],
               'precision_7' : [],
               'f1_0' : [],
               'f1_1' : [],
               'f1_2' : [],
               'f1_3' : [],
               'f1_4' : [],
               'f1_5' : [],
               'f1_6' : [],
               'f1_7' : [],
               'tpr' : [],
               'fpr' : [],
               "test_loss": [],
               "test_acc": []
    }
    
    # Make sure model on target device
    model.to(device)

    # Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
      try:
        train_loss, train_acc,r0,r1,r2,r3,r4,r5,r6,r7,p0,p1,p2,p3,p4,p5,p6,p7,f0,f1,f2,f3,f4,f5,f6,f7,tpr,fpr = train_step(model=model,
                                          dataloader=train_dataloader,
                                          loss_fn=loss_fn,
                                          optimizer=optimizer,
                                          device=device)
        test_loss, test_acc = test_step(model=model,
          dataloader=test_dataloader,
          loss_fn=loss_fn,
          device=device)
        torch.cuda.empty_cache()
        if epoch % 20 == 0:
          save_model(model=model,target_dir='models/',model_name='ViT_B_16_fine_tuned.pth')
        # Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results['recall_0'].append(r0)
        results['recall_1'].append(r1)
        results['recall_2'].append(r2)
        results['recall_3'].append(r3)
        results['recall_4'].append(r4)
        results['recall_5'].append(r5)
        results['recall_6'].append(r6)
        results['recall_7'].append(r7)
        results['precision_0'].append(p0)
        results['precision_1'].append(p1)
        results['precision_2'].append(p2)
        results['precision_3'].append(p3)
        results['precision_4'].append(p4)
        results['precision_5'].append(p5)
        results['precision_6'].append(p6)
        results['precision_7'].append(p7)
        results['f1_0'].append(f0)
        results['f1_1'].append(f1)
        results['f1_2'].append(f2)
        results['f1_3'].append(f3)
        results['f1_4'].append(f4)
        results['f1_5'].append(f5)
        results['f1_6'].append(f6)
        results['f1_7'].append(f7)
        results['tpr'].append(tpr)
        results['fpr'].append(fpr)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)
      except:
        save_model(model=model,target_dir='models/',model_name='ViT_B_16_fine_tuned.pth')
        with open('metrics/metric_dict_vit_B16_finetuned_model.pkl','wb') as f:
            pickle.dump(results,f)
        print("Error training the model at epoch {}".format(epoch))
        return results
    try:
        save_model(model=model,target_dir='models/',model_name='ViT_B_16_fine_tuned.pth')
        with open('metrics/metric_dict_vit_B16_finetuned_model.pkl','wb') as f:
            pickle.dump(results,f)
    except:
        print('Could not save model')

    # Return the filled results at the end of the epochs
    return results