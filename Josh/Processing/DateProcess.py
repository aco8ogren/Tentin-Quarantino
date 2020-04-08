def DateProcess(Date):
    numDaysPerMonth=[31,29,31,30,31,30,31,31,30,31,30,31]
    date=Date.split('-')
    days=(int(date[0])-2020)*366+sum(numDaysPerMonth[:int(date[1])-1])+int(date[2])-1
    return days
    