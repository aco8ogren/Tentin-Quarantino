def DateProcess(Date):
    import numpy as np
    from datetime import date
    dates=np.array(Date.split('-')).astype(int)
    return (date(dates[0],dates[1],dates[2])-date(2020,1,1)).days
    
    # numDaysPerMonth=[31,29,31,30,31,30,31,31,30,31,30,31]
    # date=Date.split('-')
    # days=(int(date[0])-2020)*366+sum(numDaysPerMonth[:int(date[1])-1])+int(date[2])-1
    # return days
    