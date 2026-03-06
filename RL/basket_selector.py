# basket_selector.py

BASKETS = {
    "Small":  (300,300,200),
    "Medium": (400,400,250),
    "Large":  (500,500,300)
}

def choose_basket(objects):

    total_volume = 0

    for obj in objects:
        x,y,z = obj["size"]
        total_volume += x*y*z

    for name,size in BASKETS.items():

        bx,by,bz = size

        if total_volume < bx*by*bz*0.6:
            return name,size

    return "Large",BASKETS["Large"]