# metric_2 = {"trigger classification": [0.1, 0.2, 0.3],
#             "argument classification": [0.1, 0.2, 0.3],
#             "trigger identification": [0.1, 0.2, 0.3],
#             "argument identification": [0.1, 0.2, 0.3]}
#
# epoch = 1
# for name, value in metric_2.items():
#     print(name)
#     print(value)
#     precison, recall, f1 = value
#     print(f'{name} precsion', precison, epoch)
#     print(f'{name} recall', recall, epoch)
#     print(f'{f1} f1', f1, epoch)

# ls1=[[1],[2]]
# ls2=[[3],[4]]
# ls1.extend(ls2)
# print(ls1)

ls1=[1,2]
ls2=[[3],[4]]
ls1.append(ls2)
print(ls1)

# for(int idx=0,i<10;i++){
#     print(idx)
# }

import torch
a = torch.linspace(1,12,steps=12).view(2,3,2)
print(a)
print(torch.mean(a,1))
print(a.sum(1))

ls1 = [
    [2,2,2,2],
    [2,2,2,3],
    [2,2,3,3],
    [2,3,3,3],
    [3,3,3,3]
]

ls2 = [
    [1,1,1,1],
    [1,1,1,0],
    [1,1,0,0],
    [1,0,0,0],
    [0,0,0,0]
]

t1 = torch.tensor(ls1)
t2 = torch.tensor(ls2)
t3 = t1*t2
print(t3)
t4 = t3.sum(1)
print(t4)

t5 = [4,3,2,1,0]
t6 = torch.zeros(5,4)
print(t6)
ls7 = []
for i in t5:
    tmp = [[1]*3] * i + [[0]*3] * (4-i)
    ls7.append(tmp)

print(ls7)

print(max(t5))
