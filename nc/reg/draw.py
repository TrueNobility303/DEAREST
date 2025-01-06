import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from pickle import TRUE
import numpy as np

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

out_fname_DEAREST = './result_DEAREST.csv'
out_fname_DESTRESS = './result_DESTRESS.csv'
out_fname_DSGD = './result_DSGD.csv'

# draw pictures
data_DEAREST = pd.read_csv(out_fname_DEAREST, names=["grad", "comp", "comm", "time", "loss", "gnorm", "bnorm", "mnorm"])
data_DESTRESS = pd.read_csv(out_fname_DESTRESS, names=["grad", "comp", "comm", "time", "loss", "gnorm", "bnorm", "mnorm"])
data_DSGD = pd.read_csv(out_fname_DSGD, names=["grad", "comp", "comm", "time", "loss", "gnorm", "bnorm", "mnorm"])

plt.rc('font', size=18)
plt.figure()
plt.plot(data_DSGD["comp"], data_DSGD["bnorm"], '-r', label='DSGD', marker='x', markersize=12, markerfacecolor='none',
         markevery=15, linewidth=3)
plt.plot(data_DESTRESS["comp"], data_DESTRESS["bnorm"], '-b', label='DESTRESS', marker='o', markersize=12,
         markerfacecolor='none',
         markevery=15, linewidth=3)
plt.plot(data_DEAREST["comp"], data_DEAREST["bnorm"], '-k', label=r'DEAREST${}^+$', markersize=12,
         markerfacecolor='none',
         markevery=5, linewidth=3)

plt.grid()
plt.legend(fontsize=18, frameon=True, loc='lower right')
plt.tick_params('x', labelsize=18)
plt.tick_params('y', labelsize=18)
plt.xlim(0, 5000)
plt.xlabel('#Computation')
plt.ylabel('Grad. Norm')
plt.ticklabel_format(style='sci', scilimits=(0, 0))
plt.tight_layout()
# plt.subplots_adjust(left=0.2,right=0.90,hspace=0,wspace=0)
plt.savefig('./img/' + 'reg' + '.comp.png')
plt.savefig('./img/' + 'reg' + '.comp.pdf', format='pdf')

plt.rc('font', size=18)
plt.figure()
plt.plot(data_DSGD["grad"], data_DSGD["bnorm"], '-r', label='DSGD', marker='x', markersize=12, markerfacecolor='none',
         markevery=15, linewidth=3)
plt.plot(data_DESTRESS["grad"], data_DESTRESS["bnorm"], '-b', label='DESTRESS', marker='o', markersize=12,
         markerfacecolor='none',
         markevery=15, linewidth=3)
plt.plot(data_DEAREST["grad"], data_DEAREST["bnorm"], '-k', label=r'DEAREST${}^+$', markersize=12,
         markerfacecolor='none',
         markevery=5, linewidth=3)
plt.grid()
plt.legend(fontsize=18, frameon=True, loc='lower right')
plt.tick_params('x', labelsize=18)
plt.tick_params('y', labelsize=18)
plt.xlim(0, 100000)
plt.xlabel('#LIFO')
plt.ylabel('Grad. Norm')
plt.ticklabel_format(style='sci', scilimits=(0, 0))
plt.tight_layout()
# plt.subplots_adjust(left=0.2,right=0.90,hspace=0,wspace=0)
plt.savefig('./img/' + 'reg' + '.grad.png')
plt.savefig('./img/' + 'reg' + '.grad.pdf', format='pdf')

plt.rc('font', size=18)
plt.figure()
plt.plot(data_DSGD["comm"], data_DSGD["bnorm"], '-r', label='DSGD', marker='x', markersize=12, markerfacecolor='none',
          markevery=14, linewidth=3)
plt.plot(data_DESTRESS["comm"], data_DESTRESS["bnorm"], '-b', label='DESTRESS', marker='o', markersize=12,
         markerfacecolor='none',
         markevery=17, linewidth=3)
plt.plot(data_DEAREST["comm"], data_DEAREST["bnorm"], '-k', label=r'DEAREST${}^+$',  markersize=12,
         markerfacecolor='none',
         markevery=5, linewidth=3)
plt.grid()
plt.legend(fontsize=18, frameon=True, loc='lower right')
plt.tick_params('x', labelsize=18)
plt.tick_params('y', labelsize=18)
plt.xlim(0, 1000)
plt.xlabel('#Communication')
plt.ylabel('Grad. Norm')
plt.ticklabel_format(style='sci', scilimits=(0, 0))
plt.tight_layout()
# plt.subplots_adjust(left=0.2,right=0.90,hspace=0,wspace=0)
plt.savefig('./img/' + 'reg' + '.comm.png')
plt.savefig('./img/' + 'reg' + '.comm.pdf', format='pdf')

# plt.rc('font', size=18)
# plt.figure()
# plt.plot(data_DSGD["time"], data_DSGD["bnorm"], '-r', label='DSGD', marker='x', markersize=12, markerfacecolor='none',
#           markevery=14, linewidth=3)
# # plt.plot(data_GTSARAH["time"], data_GTSARAH["bnorm"], '-r', label='GT-SARAH', marker='d', markersize=12, markerfacecolor='none',
# #           markevery=15, linewidth=3)
# plt.plot(data_DESTRESS["time"], data_DESTRESS["bnorm"], '-b', label='DESTRESS', marker='o', markersize=12,
#          markerfacecolor='none',
#          markevery=12, linewidth=3)
# plt.plot(data_DEAREST["time"], data_DEAREST["bnorm"], '-k', label='DEAREST', markersize=12,
#          markerfacecolor='none',
#          markevery=5, linewidth=3)
# plt.grid()
# plt.legend(fontsize=18, frameon=True, loc='upper right')
# plt.tick_params('x', labelsize=18)
# plt.tick_params('y', labelsize=18)
# # plt.xlim(0, 0.05)
# plt.ylabel('Grad. Norm')
# plt.xlabel('Time (s)')
# plt.tight_layout()
# plt.subplots_adjust(left=0.2,right=0.90,hspace=0,wspace=0)
# plt.savefig('./img/' + 'reg' + '.time.png')
# plt.savefig('./img/' + 'reg' + '.time.pdf', format='pdf')