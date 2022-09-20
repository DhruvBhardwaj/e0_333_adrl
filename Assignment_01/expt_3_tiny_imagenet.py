import torch
import time
import datetime

import config_tiny_imagenet as cfg

import utils as util

from datasets import getDataloader
from vae_2_CNN import VQ_Varn_AE
#########################LOGGER#########################
import sys

class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    
    def flush(self):
        pass

sys.stdout = Logger(cfg.SAVE_PATH + 'expt_1_tinyimagenet.txt')
#########################################################3

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(device)
#########################################################3
def train_VAE():
    print('-' * 59)
    torch.cuda.empty_cache()
    vae_model = VQ_Varn_AE(cfg.INPUT_H*cfg.INPUT_W, cfg.EMBED_DIM, cfg.INPUT_CH, cfg.DATA_VAR, cfg.NUM_EMBEDDINGS, cfg.BETA)
    vae_model.to(device)    
    vae_model.train()
    
    optimizer = torch.optim.Adam(vae_model.parameters(), lr=cfg.LEARN_RATE)

    data = getDataloader(cfg.DATA_PATH,cfg.BATCH_SIZE, cfg.FILE_EXTN)
    N = len(data)*cfg.BATCH_SIZE
    print('-' * 59)
    print("Starting Training of model")
    epoch_times = []

    for epoch in range(1,cfg.EPOCHS+1):        
        start_time = time.process_time()        
        total_loss = 0.0
        mse_loss = 0.0
        vq = 0.0
        
        counter = 0        
        for image_batch in data:
            #print(image_batch.size())
            #print(torch.var(image_batch))            
            counter += 1            
            optimizer.zero_grad()
            x_hat, vq_loss = vae_model.forward(image_batch.to(device))                     
            loss, m = vae_model.criterion(image_batch.to(device), x_hat, vq_loss)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            mse_loss += m.item()
            vq += vq_loss.item()            
            
            if counter%500 == 0:                
                print("Epoch {}......Step: {}/{}....... Loss={:12.5} (MSE Loss = {:12.5}, EMB + COMT Loss= {:12.5})"
                .format(epoch, counter, len(data), total_loss/N,mse_loss/N,vq/N))
        
        current_time = time.process_time()
        print("Epoch {}/{} Done, Loss = {:12.5} (MSE Loss = {:12.5}, EMB + COMT Loss= {:12.5})"
        .format(epoch, cfg.EPOCHS, total_loss/N,mse_loss/N,vq/N))

        print("Total Time Elapsed={:12.5} seconds".format(str(current_time-start_time)))

        if((mse_loss/N) < 6000):
            samples = util.return_random_batch_from_dir(cfg.DATA_PATH, cfg.FILE_EXTN, cfg.NUM_GENERATE_SAMPLES)
            r_samples = vae_model.sample(samples.to(device), cfg.NUM_GENERATE_SAMPLES,device)
            util.save_image_to_file(121,r_samples, cfg.SAVE_PATH)
        # if((mse_loss/N) < 6000):      
        #     torch.save(vae_model, cfg.SAVE_PATH + 'MODEL_E' + str(epoch) + datetime.date.today().strftime("%B %d, %Y") + '.pth')
        
        epoch_times.append(current_time-start_time)
        print('-' * 59)

    print("Total Training Time={:12.5} seconds".format(str(sum(epoch_times))))
    return vae_model

if __name__ == '__main__':
    model = train_VAE()