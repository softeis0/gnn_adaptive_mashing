from tqdm import tqdm

li = list(range(20000))

def func(x):
    return x**x

newList = []
#for x in tqdm(li):
  #  newList.append(func(x))

for x in tqdm(range(1)):
    list(map(func, li))