import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from pickle import TRUE
import numpy as np
from matplotlib.ticker import ScalarFormatter, FuncFormatter

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


out_fname_DSGD = './result_DSGD.csv'
out_fname_DRONE = './result_DRONE.csv'
out_fname_DEAREST = './result_DEAREST.csv'

# draw pictures
data_DEAREST = pd.read_csv(out_fname_DEAREST, names=["grad", "comp", "comm", "time", "loss", "gnorm", "bnorm", "mnorm"])
data_DRONE = pd.read_csv(out_fname_DRONE, names=["grad", "comp", "comm", "time", "loss", "gnorm", "bnorm", "mnorm"])
data_DSGD = pd.read_csv(out_fname_DSGD, names=["grad", "comp", "comm", "time", "loss", "gnorm", "bnorm", "mnorm"])

round_DEAREST = np.array( [i for i in range(len(data_DEAREST['loss']))])
round_DESTRESS = np.array( [i for i in range(len(data_DRONE['loss']))])
round_GTSARAH = np.array( [i for i in range(len(data_DSGD['loss']))])

plt.rc('font', size=18)
plt.figure()
plt.plot(data_DSGD["comp"], data_DSGD["loss"], '-r', label='DSGD', marker='x', markersize=12, markerfacecolor='none',
         markevery=10)
plt.plot(data_DRONE["comp"], data_DRONE["loss"], '-b', label='DRONE', marker='o', markersize=12,
         markerfacecolor='none',
         markevery=15, linewidth=3)
plt.plot(data_DEAREST['comp'], data_DEAREST["loss"], '-k', label=r'DEAREST${}^+$', markersize=12,
         markerfacecolor='none',
         markevery=3, linewidth=3)
plt.grid()
plt.legend(fontsize=18, frameon=True, loc='lower right')
plt.tick_params('x', labelsize=18)
plt.tick_params('y', labelsize=18)
plt.xlabel('#Computation')
plt.ylabel('Loss')
# plt.yticks([0.0152, 0.0153])
# plt.xticks([0, 10000])
plt.xlim(0, 1000)
plt.ticklabel_format(style='sci', scilimits=(0, 0))
# plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.3f}'))
# plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.1f}'))
plt.tight_layout()
# plt.subplots_adjust(left=0.23,right=0.90,hspace=0,wspace=0)
plt.savefig('./img/' + 'lb' + '.comp.png')
plt.savefig('./img/' + 'lb' + '.comp.pdf', format='pdf')

plt.rc('font', size=18)
plt.figure()

plt.plot(data_DSGD["grad"], data_DSGD["loss"], '-r', label='DSGD', marker='x', markersize=12, markerfacecolor='none',
         markevery=10)
plt.plot(data_DRONE["grad"], data_DRONE["loss"], '-b', label='DRONE', marker='o', markersize=12,
         markerfacecolor='none',
         markevery=20, linewidth=3)
plt.plot(data_DEAREST["grad"], data_DEAREST["loss"], '-k', label=r'DEAREST${}^+$', 
         markerfacecolor='none',
         markevery=12, linewidth=3)
plt.grid()
plt.legend(fontsize=18, frameon=True, loc='lower right')
plt.tick_params('x', labelsize=18)
plt.tick_params('y', labelsize=18)
# plt.xticks([0, 100000])
plt.xlim(0, 10000)
plt.xlabel('#LIFO')
plt.ylabel('Loss')
# plt.yticks([0.0152, 0.0153])
plt.ticklabel_format(style='sci', scilimits=(0, 0))
# formatter = FuncFormatter(lambda x, _: f'{x:.2f}')
# plt.gca().xaxis.set_major_formatter(formatter)
# formatter_x = FuncFormatter(lambda x, _: f'{x:.1f}')
# plt.gca().yaxis.set_major_formatter(formatter_x)
# plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.3f}'))
# plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.1f}'))
plt.tight_layout()
# plt.subplots_adjust(left=0.23,right=0.90,hspace=0,wspace=0)
plt.savefig('./img/' + 'lb' + '.grad.png')
plt.savefig('./img/' + 'lb' + '.grad.pdf', format='pdf')

plt.rc('font', size=18)
plt.figure()
plt.plot(data_DSGD["comm"], data_DSGD["loss"], '-r', label='DSGD', marker='x', markersize=12, markerfacecolor='none',
         markevery=20)
plt.plot(data_DRONE["comm"], data_DRONE["loss"], '-b', label='DRONE', marker='o', markersize=12,
         markerfacecolor='none',
         markevery=20, linewidth=3)
plt.plot(data_DEAREST["comm"], data_DEAREST["loss"], '-k', label=r'DEAREST${}^+$', 
         markerfacecolor='none',
         markevery=20, linewidth=3)
plt.grid()
plt.legend(fontsize=18, frameon=True, loc='lower right')
plt.tick_params('x', labelsize=18)
plt.tick_params('y', labelsize=18)
plt.xlabel('#Communication')
plt.ylabel('Loss')
# plt.yticks([0.0152, 0.0153])
plt.ticklabel_format(style='sci', scilimits=(0, 0))
plt.tight_layout()
# plt.subplots_adjust(left=0.23,right=0.90,hspace=0,wspace=0)
plt.savefig('./img/' + 'lb' + '.comm.png')
plt.savefig('./img/' + 'lb' + '.comm.pdf', format='pdf')

plt.rc('font', size=18)
plt.figure()
plt.plot(data_DSGD["time"], data_DSGD["loss"], '-r', label='DSGD', marker='x', markersize=12, markerfacecolor='none',
         markevery=10)
plt.plot(data_DRONE["time"], data_DRONE["loss"], '-b', label='DRONE', marker='o', markersize=12,
         markerfacecolor='none',
         markevery=20, linewidth=3)
plt.plot(data_DEAREST["time"], data_DEAREST["loss"], '-k', label=r'DEAREST${}^+$', 
         markerfacecolor='none',
         markevery=5, linewidth=3)
plt.grid()
plt.legend(fontsize=18, frameon=True, loc='lower right')
plt.tick_params('x', labelsize=18)
plt.tick_params('y', labelsize=18)
# plt.xlim(0, 20)
plt.xlabel('Time (s)')
plt.ylabel('Loss')
# plt.yticks([0.0152, 0.0153])
plt.ticklabel_format(style='sci', scilimits=(0, 0))
plt.tight_layout()
# plt.subplots_adjust(left=0.23,right=0.90,hspace=0,wspace=0)
plt.savefig('./img/' + 'lb' + '.time.png')
plt.savefig('./img/' + 'lb' + '.time.pdf', format='pdf')

