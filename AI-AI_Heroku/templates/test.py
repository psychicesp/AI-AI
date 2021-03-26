#%%

def solution(A):
    maximum = 0
    for i in A:
        try:
            int(i)
            if i> maximum and abs(i) < 10:
                maximum = i
        except:
            pass
    return maximum



solution([x for x in range(-20,20)])
# %%
