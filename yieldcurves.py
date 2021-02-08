import pandas as pd
import datetime as dt
from datetime import date
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la


bonds = pd.read_csv("/Users/jeremybasso/Desktop/School/APM466/Yield Curve Data.csv") #load bond data
bonds_used = [6, 9, 14, 16, 18, 20, 23, 24, 27, 28] #indexes of bonds used for bootstrap in chronological order
first = '1/18/2021'

def days_btwn(start, end):
    date1 = date(int(start.split('/')[-1]), int(start.split('/')[-3]), int(start.split('/')[-2]))
    date2 = date(int(end.split('/')[-1]), int(end.split('/')[-3]), int(end.split('/')[-2]))
    return float((date2 - date1).days)

def yrs_between(start, end):
    date1 = date(int(start.split('/')[-1]), int(start.split('/')[-3]), int(start.split('/')[-2]))
    date2 = date(int(end.split('/')[-1]), int(end.split('/')[-3]), int(end.split('/')[-2]))
    diff = ((date2.year - date1.year) * 12 + (date2.month - date1.month)) / 12
    return float(diff)

def dtm(start, bond_indices):
    days = []
    for i in bond_indices:
        ydm = bonds.iloc[i, 5].split('/')
        days.append(days_btwn(start, str(ydm[0]+'/'+ydm[1]+'/'+ydm[2])))

    return days

def dtm_months(date, bond_indices):
    months = []
    for x in dtm(date, bonds_used):
        months.append(x/30)

    return months

def get_accrued(start_date, bond_indices, i):
    bond = bonds.iloc[bond_indices[i]]
    term = days_btwn(start_date, bond[5])

    n = days_btwn(start_date, bond[5])
    while n > 0:
        n -= 180

    n = -n

    accrued = (n / 360) * float(bond[2].strip('%'))

    return accrued

def get_rhs(start_date, bond_indices, i, y):
    bond = bonds.iloc[bond_indices[i]]
    rhs = 0
    term = days_btwn(start_date, bond[5])
    k = 1
    for t in range(int((term // 180)-1)):
        rhs += (float(bond[2].strip('%')) / 2) / ((1 + (y/2)) ** ((term - (180 * (k))*2) / 360))
        k += 1
    rhs += (((float(bond[2].strip('%')) / 2) + 100)/ (1 + (y/2)) ** (term * 2 / 360))

    return rhs

def lin_inter(x, x0, x1, y0, y1):
    y = y0 + ((x - x0) * (y1-y0) / (x1-x0))
    return y

def spot(start_date, bond_indices):
    r = []
    terms = []
    r_start = '2/1/2021' #as we were instructed to round to the nearest month

    for i in range(5):
        bond = bonds.iloc[bond_indices[i]]
        term = yrs_between(r_start, bond["Maturity Date"])
        terms.append(term) #store all the terms for interpolation purposes
        price = bond[start_date]
        coup = float(bond["Coupon"].strip("%")) / 2

        #for bonds where interpolation is unneccesary
        if term % .5 == 0:
                num_coup = int(((term) // .5) - 1)
                denom = price

                t_i = .5
                for t in range(num_coup):
                    denom -= coup / ((1+(r[t])/2) ** (2*t_i))
                    t_i += .5

                y = ((((coup + 100) / (denom)) ** (1/(2*term))) - 1) * 2
                r.append(y)

        #for bonds where "multiple interpolation" is necessary
        if round(terms[i], 4) - round(terms[i-1], 4) < .5 and len(terms) > 1:
            inter_r = []
            inter_terms = []
            j = terms[-1]
            while j > 0:
                inter_terms.insert(0, j)
                j -= .5

            #populate the interpolated interest rates
            inter_r.append(lin_inter(2*inter_terms[0], 0, 1, 0, r[0]))
            inter_r.append(lin_inter(2*inter_terms[1], 1, 2, r[0], r[1]))
            inter_r.append(lin_inter(2*inter_terms[2], 2, 3, r[1], r[2]))
            inter_r.append(lin_inter(2*inter_terms[3], 3, 4, r[2], r[3]))

            denom = price + get_accrued(r_start, bond_indices, i)

            for k in range((len(inter_terms)-1)):
                denom -= coup / ((1+inter_r[k]/2) ** (2*inter_terms[k]))

            rate1 = ((((coup + 100) / denom) ** (1/(2*inter_terms[-1]))) - 1) * 2
            r.append(rate1)

    #interpolated rates for the bond greater than 6mo away
    bond = bonds.iloc[bond_indices[5]]
    terms = [.0833333333333335, .5833333333333335, 1.0833333333333335, 1.5833333333333335, 2.0833333333333335,
             2.5833333333333335, 3.0833333333333335]
    coup = float(bond["Coupon"].strip("%")) / 2

    r_special = []
    r_special.append(lin_inter(2 * terms[0], 0, 1, 0, r[0]))
    r_special.append(lin_inter(2 * terms[1], 1, 2, r[0], r[1]))
    r_special.append(lin_inter(2 * terms[2], 2, 3, r[1], r[2]))
    r_special.append(lin_inter(2 * terms[3], 3, 4, r[2], r[3]))
    r_special.append(lin_inter(2 * terms[4], 4, 5, r[3], r[4]))

    price = bond[start_date] + get_accrued(r_start, bond_indices, 5)
    r7 = 0

    lefthand = price - coup / (1 + r_special[0] / 2) - coup / (1 + r_special[1] / 2) - coup / (1 + r_special[2] / 2) \
                - coup / (1 + r_special[3] / 2) - coup / (1 + r_special[4] / 2)
    righthand = (coup / ((1 + r_special[4]/2 + (2/3)*r7 - (2/3)*r[4])**6)) + ((coup + 100) / (1+(r7/2))**7)

    while round(lefthand, 5) != round(righthand, 5) and r7 <= 1:
        r7 += .00000001
        righthand = (coup / ((1 + r_special[4] / 2 + (2 / 3) * r7 - (2 / 3) * r[4]) ** 6)) + (
                    (coup + 100) / (1 + (r7 / 2)) ** 7)

    r6 = lin_inter(2.58, 2.33, 3.08, r_special[4], r7)

    r_special.append(r6)
    r_special.append(r7)

    #interpolate the 2.5yr and 3yr semiannual spots
    r[4] = lin_inter(2.5, 2.333, 2.583, r[4], r6)
    r.append(lin_inter(3, 2.583, 3.083, r6, r7))

    terms = [.0833333333333335, .5833333333333335, 1.0833333333333335, 1.5833333333333335, 2.0833333333333335,
     2.5833333333333335, 3.0833333333333335]
    for i in range(6, 10):
        bond = bonds.iloc[bond_indices[i]]
        term = yrs_between(r_start, bond["Maturity Date"])
        terms.append(term) #store all the terms for interpolation purposes
        price = bond[start_date] #+ get_accrued(r_start, bond_indices, i)
        coup = float(bond["Coupon"].strip("%")) / 2

        num_coup = int((term // .5) - 1)
        denom = price

        t_i = .5
        for t in range(num_coup):
            denom -= coup / ((1+(r_special[t])/2) ** (2*t_i))
            t_i += .5

        y = ((((coup + 100) / (denom)) ** (1/(2*term))) - 1) * 2
        r_special.append(y)

        r.append(lin_inter(term-.0833333333333335, term-.5833333333333335, term, r_special[-2], r_special[-1]))

    return r

def ytm(start_date, bond_indices):
    ytm = []
    terms = []
    r_start = '2/1/2021' #rounded
    for i in range(10):
        bond = bonds.iloc[bond_indices[i]]
        coup = float(bond["Coupon"].strip("%")) / 2
        term = 2 * yrs_between(r_start, bond["Maturity Date"])
        terms.append(term/2)

        price = bond[start_date] - coup + get_accrued(r_start, bond_indices, i)

        y = -.0002

        while round(get_rhs(r_start, bond_indices, i, y), 2) != round(price, 2):
            y += .00001

        ytm.append(y)

    #interpolate
    for i in range(10):
        bond = bonds.iloc[bond_indices[i]]
        term = yrs_between(r_start, bond["Maturity Date"])
        if term % .5 != 0:
            x = (5 * round(float(10*term)/5))/10

            k = 0
            while x > terms[k]:
                k += 1

            y = lin_inter(x, terms[k-1], terms[k], ytm[k-1], ytm[k])
            ytm[i] = y

    return ytm

def fwd(start_date, bond_indices):
    fwd = []
    r = spot(start_date, bond_indices)

    for i in range(4):
        root = 4
        top_exp = 2

        f = ((((1 + r[i+4]/2)**(2*top_exp) / (1 + r[1])**2) ** (1/root))-1) * 2

        fwd.append(f)
        root += 4
        top_exp +=3

    return fwd

#store fwd rates
fwd18 = fwd('1/18/2021', bonds_used)
fwd19 = fwd('1/19/2021', bonds_used)
fwd20 = fwd('1/20/2021', bonds_used)
fwd21 = fwd('1/21/2021', bonds_used)
fwd22 = fwd('1/22/2021', bonds_used)
fwd25 = fwd('1/25/2021', bonds_used)
fwd26 = fwd('1/26/2021', bonds_used)
fwd27 = fwd('1/27/2021', bonds_used)
fwd28 = fwd('1/28/2021', bonds_used)
fwd29 = fwd('1/29/2021', bonds_used)

#store ytms
ytm18 = ytm('1/18/2021', bonds_used)
ytm19 = ytm('1/19/2021', bonds_used)
ytm20 = ytm('1/20/2021', bonds_used)
ytm21 = ytm('1/21/2021', bonds_used)
ytm22 = ytm('1/22/2021', bonds_used)
ytm25 = ytm('1/25/2021', bonds_used)
ytm26 = ytm('1/26/2021', bonds_used)
ytm27 = ytm('1/27/2021', bonds_used)
ytm28 = ytm('1/28/2021', bonds_used)
ytm29 = ytm('1/29/2021', bonds_used)

ytm_data = {'X1': [], 'X2': [], 'X3': [], 'X4': [], 'X5': []}
keys = list(ytm_data)
ytms = [ytm18, ytm19, ytm20, ytm21, ytm22, ytm25, ytm26, ytm27, ytm28, ytm29]
for j in range(9):
    for i in range(5):
        k = 2*i + 1
        rj1 = ytms[j+1][k]
        rj = ytms[j][k]
        if rj1 < 0:
            rj1 = .00001
        if rj < 0:
            rj = .00001
        x = np.log(rj1 / rj)
        ytm_data[keys[i]].append(x)

fwd_data = {'X1': [], 'X2': [], 'X3': [], 'X4': []}
keys = list(fwd_data)
fwds = [fwd18, fwd19, fwd20, fwd21, fwd22, fwd25, fwd26, fwd27, fwd28, fwd29]
for j in range(9):
    for i in range(4):
        rj1 = fwds[j][i]
        rj = fwds[j+1][i]

        x = np.log(rj1 / rj)

        fwd_data[keys[i]].append(x)

ytm_ts = pd.DataFrame(ytm_data, columns=['X1', 'X2', 'X3', 'X4', 'X5'])
fwd_ts = pd.DataFrame(fwd_data, columns=['X1', 'X2', 'X3', 'X4'])

print(ytm_ts.cov())
print(fwd_ts.cov())
print(la.eig(ytm_ts.cov())[0])
print(la.eig(ytm_ts.cov())[1])
print(la.eig(fwd_ts.cov())[0])
print(la.eig(fwd_ts.cov())[1])

# plot the spot rate curve
plot1 = plt.figure(1)
time_axis = [6, 12, 18, 24, 30, 36, 42, 48, 54, 60]
plt.plot(time_axis, spot('1/18/2021', bonds_used), label='1/18/2021')
plt.plot(time_axis, spot('1/19/2021', bonds_used), label='1/19/2021')
plt.plot(time_axis, spot('1/20/2021', bonds_used), label='1/20/2021')
plt.plot(time_axis, spot('1/21/2021', bonds_used), label='1/21/2021')
plt.plot(time_axis, spot('1/22/2021', bonds_used), label='1/22/2021')
plt.plot(time_axis, spot('1/25/2021', bonds_used), label='1/25/2021')
plt.plot(time_axis, spot('1/26/2021', bonds_used), label='1/26/2021')
plt.plot(time_axis, spot('1/27/2021', bonds_used), label='1/27/2021')
plt.plot(time_axis, spot('1/28/2021', bonds_used), label='1/28/2021')
plt.plot(time_axis, spot('1/29/2021', bonds_used), label='1/29/2021')
plt.legend(loc="upper left")
plt.title('Spot Curve')
plt.ylabel('Spot Rate')
plt.xlabel('Months')

# plot the yield curve
plot2 = plt.figure(2)
plt.plot(dtm_months('1/18/2021', bonds_used), ytm18, label='1/18/2021')
plt.plot(dtm_months('1/19/2021', bonds_used), ytm19, label='1/19/2021')
plt.plot(dtm_months('1/20/2021', bonds_used), ytm20, label='1/20/2021')
plt.plot(dtm_months('1/21/2021', bonds_used), ytm21, label='1/21/2021')
plt.plot(dtm_months('1/22/2021', bonds_used), ytm22, label='1/22/2021')
plt.plot(dtm_months('1/25/2021', bonds_used), ytm25, label='1/25/2021')
plt.plot(dtm_months('1/26/2021', bonds_used), ytm26, label='1/26/2021')
plt.plot(dtm_months('1/27/2021', bonds_used), ytm27, label='1/27/2021')
plt.plot(dtm_months('1/28/2021', bonds_used), ytm28, label='1/28/2021')
plt.plot(dtm_months('1/29/2021', bonds_used), ytm29, label='1/29/2021')
plt.legend(loc="upper left")
plt.title('Yield Curve')
plt.ylabel('YTM')
plt.xlabel('Months')

# plot the forward rate curve
plot3 = plt.figure(3)
plt.plot([2, 3, 4, 5], fwd18, label='1/18/2021')
plt.plot([2, 3, 4, 5], fwd19, label='1/19/2021')
plt.plot([2, 3, 4, 5], fwd20, label='1/20/2021')
plt.plot([2, 3, 4, 5], fwd21, label='1/21/2021')
plt.plot([2, 3, 4, 5], fwd22, label='1/22/2021')
plt.plot([2, 3, 4, 5], fwd25, label='1/25/2021')
plt.plot([2, 3, 4, 5], fwd26, label='1/26/2021')
plt.plot([2, 3, 4, 5], fwd27, label='1/27/2021')
plt.plot([2, 3, 4, 5], fwd28, label='1/28/2021')
plt.plot([2, 3, 4, 5], fwd29, label='1/29/2021')
plt.legend(loc="upper left")
plt.title('1-X Yr Forward Curve')
plt.ylabel('Forward Rte')
plt.xlabel('1-X Year')

plt.show()

