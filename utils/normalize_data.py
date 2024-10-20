

def normalize_lendmarks(lendmarks):
    num = 0;
    for lendmark in lendmarks.landmark:
        print(num, ": ",lendmark.x, lendmark.y,lendmark.z)
        num += 1