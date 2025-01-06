import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from pickle import TRUE
import numpy as np
from matplotlib.ticker import (MultipleLocator, 
                               FormatStrFormatter, 
                               AutoMinorLocator) 
from matplotlib.ticker import ScalarFormatter, FuncFormatter

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

out_fname_DEAREST = './result_DEAREST.csv'
out_fname_DRONE = './result_DRONE.csv'
out_fname_DSGD = './result_DSGD.csv'

# draw pictures
data_DEAREST = pd.read_csv(out_fname_DEAREST, names=["grad", "comp", "comm", "time", "loss", "gnorm", "bnorm", "mnorm"])
data_DRONE = pd.read_csv(out_fname_DRONE, names=["grad", "comp", "comm", "time", "loss", "gnorm", "bnorm", "mnorm"])
data_DSGD = pd.read_csv(out_fname_DSGD, names=["grad", "comp", "comm", "time", "loss", "gnorm", "bnorm", "mnorm"])

plt.rc('font', size=18)
fig, ax = plt.subplots() 
ax.plot(data_DSGD["comp"][1:], data_DSGD["loss"][1:], '-r', label='DSGD', marker='x', markersize=12, markerfacecolor='none',
         markevery=20, linewidth=3)
ax.plot(data_DRONE["comp"], data_DRONE["loss"], '-b', label='DRONE', marker='o', markersize=12,
         markerfacecolor='none',
         markevery=5, linewidth=3)
ax.plot(data_DEAREST["comp"], data_DEAREST["loss"], '-k', label=r'DEAREST${}^+$', markersize=12,
         markerfacecolor='none',
         markevery=5, linewidth=3)

plt.grid()
plt.legend(fontsize=18, frameon=True, loc='lower left')
plt.tick_params('x', labelsize=18)
plt.tick_params('y', labelsize=18)
plt.xlim(0, 100000)
plt.xlabel('#Computation')
plt.ylabel('Loss')
plt.ticklabel_format(style='sci', scilimits=(0, 0))
# Custom formatter for scientific notation with 4 digits
# formatter = FuncFormatter(lambda x, _: f'{x:.2f}')
# plt.gca().yaxis.set_major_formatter(formatter)
plt.tight_layout()
plt.subplots_adjust(left=0.2,right=0.90,hspace=0,wspace=0)
plt.savefig('./img/' + 'rcv1' + '.comp.png')
plt.savefig('./img/' + 'rcv1' + '.comp.pdf', format='pdf')

plt.rc('font', size=18)
plt.figure()
plt.plot(data_DSGD["grad"][1:], data_DSGD["loss"][1:], '-r', label='DSGD', marker='x', markersize=12, markerfacecolor='none',
         markevery=20, linewidth=3)
plt.plot(data_DRONE["grad"], data_DRONE["loss"], '-b', label='DRONE', marker='o', markersize=12,
         markerfacecolor='none',
         markevery=5, linewidth=3)
plt.plot(data_DEAREST["grad"], data_DEAREST["loss"], '-k', label=r'DEAREST${}^+$', markersize=12,
         markerfacecolor='none',
         markevery=5, linewidth=3)
plt.grid()
plt.legend(fontsize=18, frameon=True, loc='lower left')
plt.tick_params('x', labelsize=18)
plt.tick_params('y', labelsize=18)
plt.xlim(0, 1000000)
plt.xlabel('#LIFO')
plt.ylabel('Loss')
plt.ticklabel_format(style='sci', scilimits=(0, 0))
plt.tight_layout()
# plt.subplots_adjust(left=0.2,right=0.90,hspace=0,wspace=0)
plt.savefig('./img/' + 'rcv1' + '.grad.png')
plt.savefig('./img/' + 'rcv1' + '.grad.pdf', format='pdf')

plt.rc('font', size=18)
plt.figure()
plt.plot(data_DSGD["comm"][1:], data_DSGD["loss"][1:], '-r', label='DSGD', marker='x', markersize=12, markerfacecolor='none',
         markevery=14, linewidth=3)
plt.plot(data_DRONE["comm"], data_DRONE["loss"], '-b', label='DRONE', marker='o', markersize=12,
         markerfacecolor='none',
         markevery=17, linewidth=3)
plt.plot(data_DEAREST["comm"], data_DEAREST["loss"], '-k', label=r'DEAREST${}^+$',  markersize=12,
         markerfacecolor='none',
         markevery=5, linewidth=3)
plt.grid()
plt.legend(fontsize=18, frameon=True, loc='lower left')
plt.tick_params('x', labelsize=18)
plt.tick_params('y', labelsize=18)
plt.xlim(0, 2000)
plt.xlabel('#Communication')
plt.ylabel('Loss')
plt.ticklabel_format(style='sci', scilimits=(0, 0))
plt.tight_layout()
# plt.subplots_adjust(left=0.2,right=0.90,hspace=0,wspace=0)
plt.savefig('./img/' + 'rcv1' + '.comm.png')
plt.savefig('./img/' + 'rcv1' + '.comm.pdf', format='pdf')

plt.rc('font', size=18)
plt.figure()
plt.plot(data_DSGD["time"][1:], data_DSGD["loss"][1:], '-r', label='DSGD', marker='x', markersize=12, markerfacecolor='none',
         markevery=14, linewidth=3)
plt.plot(data_DRONE["time"], data_DRONE["loss"], '-b', label='DRONE', marker='o', markersize=12,
         markerfacecolor='none',
         markevery=12, linewidth=3)
plt.plot(data_DEAREST["time"], data_DEAREST["loss"], '-k', label=r'DEAREST${}^+$', markersize=12,
         markerfacecolor='none',
         markevery=5, linewidth=3)
plt.grid()
plt.legend(fontsize=18, frameon=True, loc='lower left')
plt.tick_params('x', labelsize=18)
plt.tick_params('y', labelsize=18)
plt.xlim(0, 100)
plt.ylabel('Loss')
plt.xlabel('Time (s)')
plt.ticklabel_format(style='sci', scilimits=(0, 0))
plt.tight_layout()
# plt.subplots_adjust(left=0.2,right=0.90,hspace=0,wspace=0)
plt.savefig('./img/' + 'rcv1' + '.time.png')
plt.savefig('./img/' + 'rcv1' + '.time.pdf', format='pdf')