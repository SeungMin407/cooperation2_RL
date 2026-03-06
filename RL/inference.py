# inference.py

import torch

from packing_env import PackingEnv
from basket_selector import choose_basket
from policy_network import PolicyNetwork


model=PolicyNetwork(20,3)

model.load_state_dict(torch.load("packing_model.pth"))

model.eval()


objects=[

{"size":(60,50,40),"durability":2},
{"size":(30,40,20),"durability":4},
{"size":(50,50,50),"durability":1}

]

basket_name,basket_size=choose_basket(objects)

env=PackingEnv(basket_size)

env.set_objects(objects)

state=env.reset()

done=False

result=[]


while not done:

    heightmap=torch.tensor(env.heightmap).float().unsqueeze(0).unsqueeze(0)

    features=torch.tensor(state).float().unsqueeze(0)

    action=model(heightmap,features).detach().numpy()[0]

    next_state,reward,done,_=env.step(action)

    result.append({
        "object":env.index,
        "pos":action.tolist()
    })

    state=next_state


print(result)