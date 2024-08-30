import pickle
result=pickle.load(open('./checkpoints/cifar10/eval_result.pkl','rb'))
print(result)
print(len(result))

#格式为{'clean_acc': tensor(94.1700, device='cuda:0'), 'bd_acc': tensor(98.7700, device='cuda:0'), 'cross_acc': tensor(92.1400, device='cuda:0')}，绘制训练图
import matplotlib.pyplot as plt
import numpy as np
clean_acc=[]
bd_acc=[]
cross_acc=[]
for i in range(len(result)):
    clean_acc.append(result[i]['clean_acc'].item())
    bd_acc.append(result[i]['bd_acc'].item())
    cross_acc.append(result[i]['cross_acc'].item())
x = np.arange(len(clean_acc))
#标题WaNet模型训练图
plt.title('BppAttack model training')
plt.plot(x,clean_acc,label='clean_acc')
plt.plot(x,bd_acc,label='bd_acc')
plt.plot(x,cross_acc,label='cross_acc')
plt.legend()
plt.show()
