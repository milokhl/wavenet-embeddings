from matplotlib import pyplot as plt

def ShowPlotExample():
  fig, axs = plt.subplots(3, 1, figsize=(10, 7))
  axs[0].plot(enc1[0]); 
  axs[0].set_title('Encoding 1')
  axs[1].plot(enc2[0]);
  axs[1].set_title('Encoding 2')
  axs[2].plot(enc_mix[0]);
  axs[2].set_title('Average')
