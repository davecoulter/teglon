import numpy as np
from scipy.misc import factorial


def binomial(yes, no):
    yes = float(yes)
    no = float(no)

    if yes + no < 170:
        temp = binstat(yes + no, yes)
        errl = temp[1] - temp[0]
        erru = temp[2] - temp[1]

        if (yes / (yes + no) == 1):
            erru = 0
        if (yes / (yes + no) == 0): 
            errl = 0

    if (yes + no >= 170):
        if yes == 0:
            erru = 1. / np.sqrt(yes + no)
            errl = 0.
        else:

            eb = yes / (yes + no)

            erru = eb * (1. / yes - 1. / (yes + no)) ** .5
            errl = eb * (1. / yes - 1. / (yes + no)) ** .5

    err = [errl, erru]
    return err


def binstat(n, c):
    nsamp = 1000
    if (n > 170):
        print('sorry, too large of a sample, factorial blows up')
        return 0

    if (c > n):
        print('sample size must be larger than selected number')
        return 0

    sig = 1

    ebm = 1. * c / n
    eb = np.arange(nsamp + 1) / (1. * nsamp)

    p2 = eb * 0.
    for i in np.arange(c + 1): p2 = p2 + (factorial(n + 1) / (factorial(i) * factorial(n + 1 - i))) * eb ** i * (
            1. - eb) ** (n + 1 - i)
    tmp = p2 - 0.841345  # gauss_pdf(sig)

    w = np.where(tmp <= 0)[0]
    cnt = len(w)
    ebl = eb[w[0]]
    tmp = p2 - 1 + 0.841345  # gauss_pdf(sig)
    w = np.where(tmp >= 0)[0]
    cnt = len(w)
    if cnt:
        ebu = eb[w[cnt - 1]]
    else:
        ebu = 0
    return [ebl, ebm, ebu]