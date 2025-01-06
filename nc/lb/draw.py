import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from pickle import TRUE
import numpy as np

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


out_fname_DSGD = './result_DSGD.csv'
out_fname_DESTRESS = './result_DESTRESS.csv'
out_fname_DEAREST = './result_DEAREST.csv'

# draw pictures
data_DEAREST = pd.read_csv(out_fname_DEAREST, names=["grad", "comp", "comm", "time", "loss", "gnorm", "bnorm", "mnorm"])
data_DESTRESS = pd.read_csv(out_fname_DESTRESS, names=["grad", "comp", "comm", "time", "loss", "gnorm", "bnorm", "mnorm"])
data_DSGD = pd.read_csv(out_fname_DSGD, names=["grad", "comp", "comm", "time", "loss", "gnorm", "bnorm", "mnorm"])

round_DEAREST = np.array( [i for i in range(len(data_DEAREST['loss']))])
round_DESTRESS = np.array( [i for i in range(len(data_DESTRESS['loss']))])
round_GTSARAH = np.array( [i for i in range(len(data_DSGD['loss']))])

plt.rc('font', size=18)
plt.figure()
plt.plot(data_DSGD["comp"], data_DSGD["mnorm"], '-r', label='DSGD', marker='x', markersize=12, markerfacecolor='none',
         markevery=3)
plt.plot(data_DESTRESS["comp"], data_DESTRESS["mnorm"], '-b', label='DESTRESS', marker='o', markersize=12,
         markerfacecolor='none',
         markevery=15, linewidth=3)
plt.plot(data_DEAREST['comp'], data_DEAREST["mnorm"], '-k', label=r'DEAREST${}^+$', markersize=12,
         markerfacecolor='none',
         markevery=3, linewidth=3)
plt.grid()
plt.legend(fontsize=18, frameon=True, loc='lower right')
plt.tick_params('x', labelsize=18)
plt.tick_params('y', labelsize=18)
plt.xlim(0, 400)
plt.ylim(data_DEAREST["mnorm"].iloc[30], )
plt.ticklabel_format(style='sci', scilimits=(0, 0))
plt.xlabel('#Computation')
plt.ylabel('Grad. Norm')
plt.ticklabel_format(style='sci', scilimits=(0, 0))
plt.tight_layout()
plt.savefig('./img/' + 'lb' + '.comp.png')
plt.savefig('./img/' + 'lb' + '.comp.pdf', format='pdf')

plt.rc('font', size=18)
plt.figure()

plt.plot(data_DSGD["grad"], data_DSGD["mnorm"], '-r', label='DSGD', marker='x', markersize=12, markerfacecolor='none',
         markevery=2)
plt.plot(data_DESTRESS["grad"], data_DESTRESS["mnorm"], '-b', label='DESTRESS', marker='o', markersize=12,
         markerfacecolor='none',
         markevery=20, linewidth=3)
# plt.plot(data_DGD["grad"], data_DGD["gnorm"], '-c', label='DESTRESS+', marker='x', markersize=12,
#          markerfacecolor='none',
#          markevery=14, linewidth=3)
plt.plot(data_DEAREST["grad"], data_DEAREST["mnorm"], '-k', label=r'DEAREST${}^+$', 
         markerfacecolor='none',
         markevery=12, linewidth=3)
plt.grid()
plt.legend(fontsize=18, frameon=True, loc='lower right')
plt.tick_params('x', labelsize=18)
plt.tick_params('y', labelsize=18)
# plt.xlim(0, data_DEAREST["grad"].iloc[-1])
# plt.ylim(data_DEAREST["mnorm"].iloc[-1], 0.18)
#plt.ticklabel_format(style='sci', scilimits=(0, 0))
#plt.xlim(0, data_GTSARAH["grad"].iloc[-1])
plt.xlim(0, 3500)
plt.ylim(data_DEAREST["mnorm"].iloc[30], )
plt.xlabel('#LIFO')
plt.ylabel('Grad. Norm')
plt.ticklabel_format(style='sci', scilimits=(0, 0))
plt.tight_layout()
plt.savefig('./img/' + 'lb' + '.grad.png')
plt.savefig('./img/' + 'lb' + '.grad.pdf', format='pdf')

plt.rc('font', size=18)
plt.figure()
plt.plot(data_DSGD["comm"], data_DSGD["mnorm"], '-r', label='DSGD', marker='x', markersize=12, markerfacecolor='none',
         markevery=20)
plt.plot(data_DESTRESS["comm"], data_DESTRESS["mnorm"], '-b', label='DESTRESS', marker='o', markersize=12,
         markerfacecolor='none',
         markevery=20, linewidth=3)
plt.plot(data_DEAREST["comm"], data_DEAREST["mnorm"], '-k', label=r'DEAREST${}^+$', 
         markerfacecolor='none',
         markevery=20, linewidth=3)
plt.grid()
plt.legend(fontsize=18, frameon=True, loc='lower right')
plt.tick_params('x', labelsize=18)
plt.tick_params('y', labelsize=18)
plt.xlabel('#Communication')
plt.ylabel('Grad. Norm')
plt.xlim(0,500)
plt.ylim(data_DEAREST["mnorm"].iloc[30], )
plt.ticklabel_format(style='sci', scilimits=(0, 0))
plt.tight_layout()
plt.savefig('./img/' + 'lb' + '.comm.png')
plt.savefig('./img/' + 'lb' + '.comm.pdf', format='pdf')

plt.rc('font', size=18)
plt.figure()
plt.plot(data_DSGD["time"], data_DSGD["mnorm"], '-r', label='DSGD', marker='x', markersize=12, markerfacecolor='none',
         markevery=10)
plt.plot(data_DESTRESS["time"], data_DESTRESS["mnorm"], '-b', label='DESTRESS', marker='o', markersize=12,
         markerfacecolor='none',
         markevery=20, linewidth=3)
plt.plot(data_DEAREST["time"], data_DEAREST["mnorm"], '-k', label=r'DEAREST${}^+$', 
         markerfacecolor='none',
         markevery=5, linewidth=3)
plt.grid()
plt.legend(fontsize=18, frameon=True, loc='lower right')
plt.tick_params('x', labelsize=18)
plt.tick_params('y', labelsize=18)
plt.xlim(0, 10)
plt.ylim(data_DEAREST["mnorm"].iloc[30], )
# plt.ylim(data_DEAREST["mnorm"].iloc[-1], 0.18)
# plt.ticklabel_format(style='sci', scilimits=(0, 0))
plt.xlabel('Time (s)')
plt.ylabel('Grad. Norm')
plt.ticklabel_format(style='sci', scilimits=(0, 0))
plt.tight_layout()
plt.savefig('./img/' + 'lb' + '.time.png')
plt.savefig('./img/' + 'lb' + '.time.pdf', format='pdf')

