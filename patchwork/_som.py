import numpy as np
from tqdm import tqdm



def _find_BMU(SOM,x):
    """
    :SOM: (m,n,d) numpy array representing SOM weights
    :x: (d,) array containing a training example
    """
    distSq = ((SOM - x)**2).sum(axis=2)
    return np.unravel_index(np.argmin(distSq, axis=None), distSq.shape)

def _update_weights(SOM, train_ex, learn_rate, radius_sq, 
                   BMU_coord, step=3):
    """
    Update the weights of the SOM cells when given a single training example 
    
    :SOM: (m,n,d) numpy array representing SOM weights
    :train_ex:  (d,) array containing a training example
    :learn_rate: float; learning rate
    :radius_sq: float; denominator of exponent
    :BMU_coord: tuple containing coordinates of BMU
    :step: how many grid steps away from the BMU to compute weight updates for
    """
    g, h = BMU_coord
    #if radius is close to zero then only BMU is changed
    if radius_sq < 1e-3:
        SOM[g,h,:] += learn_rate * (train_ex - SOM[g,h,:])
        return SOM
    # Change all cells in a small neighborhood of BMU
    for i in range(max(0, g-step), min(SOM.shape[0], g+step)):
        for j in range(max(0, h-step), min(SOM.shape[1], h+step)):
            dist_sq = np.square(i - g) + np.square(j - h)
            dist_func = np.exp(-dist_sq / 2 / radius_sq)
            SOM[i,j,:] += learn_rate * dist_func * (train_ex - SOM[i,j,:])   
    return SOM    

def train_SOM(SOM, train_data, learn_rate = .1, radius_sq = 1, 
             lr_decay = .1, radius_decay = .1, epochs = 10):   
    """
    
    :SOM: (m,n,d) numpy array representing SOM weights
    :train_data:  (N,d) array containing a training example
    :learn_rate: float; initial learning rate
    :radius_sq: float; initial denominator of exponent
    :lr_decay: float; decay constant for learning rate
    :radius_decay: float; decay constant for radius
    :epochs: how many epochs to train for
    """
    N = train_data.shape[0]
    learn_rate_0 = learn_rate
    radius_0 = radius_sq
    for epoch in tqdm(np.arange(0, epochs)):
        train_indices = np.random.choice(np.arange(N), size=N, replace=False)
        #rand.shuffle(train_data)      
        for i in train_indices: #train_ex in train_data:
            g, h = _find_BMU(SOM, train_data[i])
            SOM = _update_weights(SOM, train_data[i], 
                                 learn_rate, radius_sq, (g,h))
        # Update learning rate and radius
        learn_rate = learn_rate_0 * np.exp(-epoch * lr_decay)
        radius_sq = radius_0 * np.exp(-epoch * radius_decay)            
    return SOM


