from greedy_packing import greedy_pack

basket = (400,400,250)

objects = [

    (60,40,30,5),
    (80,60,50,2),
    (50,50,40,4),
    (40,40,20,1)

]

result = greedy_pack(basket, objects)

print(result)
