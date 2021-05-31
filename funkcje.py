import numpy as np
import matplotlib.pyplot as plt


def DefinicjaGeometrii():
    wezly = np.array([[1, 0],
                      [2, 1],
                      [3, 0.5],
                      [4, 0.75]])
    elementy = np.array([[1, 1, 3],
                         [2, 4, 2],
                         [3, 3, 4]])
    n = np.shape(wezly)[0]

    return wezly,elementy,n

def AutomatycznaGeometria(p,k,n):
    # p - poczatek przedzialu
    # k - koniec przedzialu
    # n - liczba wezlow
    tmp = (k-p) / (n-1) # odległośc pomiedzy dwoma wezlami
    matrix = np.array([1,p]) # zapisujemy pierwszy wezel - pierwsza kolumna indeksy, druga pozycja wezla
    matrix2 = np.array([1,1,2]) # zapisujemy elementy - pierwsza indeks elementu, kolejna poczatek i koniec

    for i in range(1, n, 1):
        matrix = np.block([
            [matrix],
            [i+1, i * tmp + p],
        ])
    for i in range (2,n,1): #
        matrix2 = np.block([
            [matrix2],
            [i,i,i+1]
        ])

    return matrix, matrix2

def RysowanieGeometrii(wezly,elementy):
    y = np.zeros(wezly.shape[0])
    plt.plot(wezly[:, 1], y, marker='o')
    for i in range(0, np.size(y), 1):
             plt.text(x=wezly[i, 1], y=y[i] + 0.010, s=int(wezly[i, 0]), color='red')
    for i in range(0, np.size(y), 1):
             plt.text(x=wezly[i, 1], y=y[i] - 0.010, s=str(round(wezly[i, 1], 2)), color='green')
    for i in range(0, np.size(y) - 1, 1):
             wp = elementy[i, 1]
             wk = elementy[i, 2]
             plt.text(x=(wezly[wp-1, 1] + wezly[wk-1, 1]) / 2, y=y[i] + 0.005, s=int(i + 1), color='blue')

    plt.show()


def Aij(df_i, df_j, c, f_i, f_j):


    f_pod = lambda x: -df_i(x)*df_j(x) + c*f_i(x)*f_j(x)

    return f_pod


def Alokacja(n):

    A = np.zeros([n, n])
    b = np.zeros([n, 1])

    return A, b


def fBazowe(n):

    if n == 0:
        f = (lambda x: 0 * x+1)
        df = (lambda x: 0 * x)

    elif n == 1:

        f = (lambda x: -0.5 * x + 1 / 2, lambda x: 0.5 * x + 0.5)
        df = (lambda x: -1 / 2 + 0 * x, lambda x: 0.5 + 0 * x)

    elif n == 2:
        f = (lambda x: 0.5 * x ** 2 - 0.5 * x, lambda x: -x ** 2 + 1, lambda x: 0.5 * x ** 2 + 0.5 * x)
        df = (lambda x: x - 0.5, lambda x: -2 * x, lambda x: x + 0.5)

    else:
        raise Exception("Nastapil blad w fBazowe().")

    return f, df


def Rozwiazanie(wezly, elementy, u):

    x = wezly[:, 1]
    plt.plot(x, u, 'ro')
    RysowanieGeometrii(wezly, elementy)






