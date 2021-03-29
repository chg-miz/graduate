import torch

# t1 = torch.randn(2,3,6)
# print(t1)
# t1[0]= torch.index_select(t1[0],0,torch.tensor([0,2]))
# t1[1]= torch.index_select(t1[1],0,torch.tensor([0,2]))
# print(t1)

t1 = torch.randn(4,6)
print(t1)
t1= torch.index_select(t1,0,torch.tensor([0,2,0,0,0,0]))
print(t1)