
import multiprocessing
import pandas as pd
import numpy as np
import time

def _apply_df(args):
    df, func, num, kwargs = args
    tmp = df.apply(func, **kwargs)
    print(time.time())
    time.sleep(5)
    print(time.time())
    return num, tmp

def apply_by_multiprocessing(df,func,**kwargs):
  workers=kwargs.pop('workers')
  pool = multiprocessing.Pool(processes=workers)
  result = pool.map(_apply_df, [(d, func, i, kwargs) for i,d in enumerate(np.array_split(df, workers))])  
  pool.close()
  result=sorted(result,key=lambda x:x[0])
  return pd.concat([i[1] for i in result])

def square(x):
    return x**x
  
if __name__ == '__main__':
    df = pd.DataFrame({'a':range(10000), 'b':range(10000)})
    print(df.head(10))
    works = 2
    print(works)
    tic = time.time()
    df2 = apply_by_multiprocessing(df, square, axis=1, workers=works)  
    toc = time.time()
    print(df2.head(10))
    print('Time to execute with %d workers: %f' %(works,toc-tic))


    
# import multiprocessing
# import pandas as pd
# import numpy as np

# def _apply_df(args):
#     df, func, kwargs = args
#     return df.apply(func, **kwargs)

# def apply_by_multiprocessing(df, func, **kwargs):
#     workers = kwargs.pop('workers')
#     pool = multiprocessing.Pool(processes=workers)
#     result = pool.map(_apply_df, [(d, func, kwargs)
#             for d in np.array_split(df, workers)])
#     pool.close()
#     return pd.concat(list(result))
    
# def square(x):
#     return x**x
    
# if __name__ == '__main__':
#     df = pd.DataFrame({'a':range(10), 'b':range(10)})
#     apply_by_multiprocessing(df, square, axis=1, workers=4)  
#     ## run by 4 processors