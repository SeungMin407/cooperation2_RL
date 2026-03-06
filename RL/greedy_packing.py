# greedy_baseline.py

def greedy_pack(objects,basket):

    w,d,h=basket

    x=0
    y=0
    z=0

    results=[]

    for i,obj in enumerate(objects):

        sx,sy,sz=obj["size"]

        if x+sx>w:

            x=0
            y+=sy

        if y+sy>d:

            y=0
            z+=sz

        if z+sz>h:
            break

        results.append({

            "object_index":i,
            "pos":(x,y,z),
            "rotation":(0,0,0)

        })

        x+=sx

    return results
