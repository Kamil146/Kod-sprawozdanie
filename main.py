import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as spint
import funkcje as fun

if __name__ == '__main__':

    # Preprocessing

    ## parametry sterujace
    c = 0
    f = lambda x: 0 * x  # wymuszenie


    #RECZNE
    #WEZLY, ELEMENTY,n =  fun.DefinicjaGeometrii()


    #AUTOMATYCZNE
    n = 6
    x_a = 0
    x_b = 6


    WEZLY, ELEMENTY = fun.AutomatycznaGeometria(x_a, x_b, n)


    # warunki brzegowe
    WB = [{"ind": 1, "typ": 'D', "wartosc": 1},
          {"ind": n, "typ": 'D', "wartosc": 2}]


    fun.RysowanieGeometrii(WEZLY, ELEMENTY)

    A, b = fun.Alokacja(n)



    stopien = 1
    phi, dphi = fun.fBazowe(stopien)

    # #wyswietlenie funkcji i pochodnych
    z = np.linspace(-1,1, 101)
    plt.plot(z, phi[0](z), 'pink' )
    plt.plot(z, phi[1](z), 'gray' )
    #plt.plot(z, phi[2](z), 'orange')
    plt.show()
    plt.plot(z, dphi[0](z), 'black' )
    plt.plot(z, dphi[1](z), 'brown' )
    #plt.plot(z, dphi[2](z), 'green')
    plt.show()




    liczba_elem = np.shape(ELEMENTY)[0]
    for ee in np.arange(0, liczba_elem):
        elemGlobalInd = ELEMENTY[ee, 0]
        elem_wez_pocz = ELEMENTY[ee, 1]
        elem_wez_konc = ELEMENTY[ee, 2]
        indGlobalneWezlow = np.array([elem_wez_pocz, elem_wez_konc])
        x_a = WEZLY[elem_wez_pocz - 1, 1]
        x_b = WEZLY[elem_wez_konc - 1, 1]
        M = np.zeros([stopien + 1, stopien + 1])
        J = (x_b - x_a) / 2
        m = 0
        n = 0
        M[m, n] = J * spint.quad(fun.Aij(dphi[m], dphi[n], c, phi[m], phi[n]), -1, 1)[0]
        m = 0
        n = 1
        M[m, n] = J * spint.quad(fun.Aij(dphi[m], dphi[n], c, phi[m], phi[n]), -1, 1)[0]
        m = 1
        n = 0
        M[m, n] = J * spint.quad(fun.Aij(dphi[m], dphi[n], c, phi[m], phi[n]), -1, 1)[0]
        m = 1
        n = 1
        M[m, n] = J * spint.quad(fun.Aij(dphi[m], dphi[n], c, phi[m], phi[n]), -1, 1)[0]
        A[np.ix_(indGlobalneWezlow - 1, indGlobalneWezlow - 1)] += M



    #Uwzglednienie warunkow brzegowych

    # if WB[0]['typ'] == 'D'and WB[1]['typ'] == 'D' :
    #     ind_wezla_pocz = WB[0]['ind']
    #     ind_wezla_konc = WB[1]['ind']
    #     wart_war_brzeg_pocz = WB[0]['wartosc']
    #     wart_war_brzeg_konc = WB[1]['wartosc']
    #
    #     iwp_p = ind_wezla_pocz - 1
    #     iwp_k = ind_wezla_konc -1
    #     A=np.delete(A,[iwp_p,iwp_k],0)
    #
    #     b = np.delete(b,[iwp_p,iwp_k], 0)
    #     wb=np.shape(b)[0]
    #     for i in np.arange(0,wb):
    #         b[i]=b[i]-A[i,iwp_p]*wart_war_brzeg_pocz
    #     for j in np.arange(0,wb):
    #         b[j]=b[j]-A[j,iwp_k]*wart_war_brzeg_konc
    #     A = np.delete(A, [iwp_p, iwp_k], 1)


    if WB[0]['typ'] == 'D':
        ind_wezla = WB[0]['ind']
        wart_war_brzeg = WB[0]['wartosc']

        iwp = ind_wezla - 1

        WZMACNIACZ = 10 ** 14

        b[iwp] = A[iwp, iwp] * WZMACNIACZ * wart_war_brzeg
        A[iwp, iwp] = A[iwp, iwp] * WZMACNIACZ

    if WB[1]['typ'] == 'D':
        ind_wezla = WB[1]['ind']
        wart_war_brzeg = WB[1]['wartosc']

        iwp = ind_wezla - 1

        WZMACNIACZ = 10 ** 14

        b[iwp] = A[iwp, iwp] * WZMACNIACZ * wart_war_brzeg
        A[iwp, iwp] = A[iwp, iwp] * WZMACNIACZ



    # Rozwiazanie

    u = np.linalg.solve(A, b)
    #u=np.vstack(([wart_war_brzeg_pocz],u,[wart_war_brzeg_konc])) #przy drugim sposobie musimy uwzglednic
    print(u)
    fun.Rozwiazanie(WEZLY, ELEMENTY,u)